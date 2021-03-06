import json
import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import sklearn
import sklearn.metrics
import tensorflow as tf

from . import settings
from .plotting import plot_confusion_matrix, plot_to_image
from .settings import SEED, BaseCNN

log = logging.getLogger(settings.APP_NAME)


class TrainingRun:
    def __init__(
        self,
        name: str,
        test_freq: int,
        base_cnn: BaseCNN,
        weights: str,
        metrics: List[str],
        metric_accuracy: str,
        splits_dir: Path,
        class_names: List[str],
        min_accuracy: float,
        dataset_dir: Path,
        batch_size: int,
        max_epochs: int,
        cp_dir: Path,
        sv_dir: Path,
        py_dir: Path,
        tb_dir: Path,
        cm_dir: Path,
    ):
        # Set received instance attributes
        self.name: str = name
        self.max_epochs: int = max_epochs
        self.test_freq: int = test_freq
        self.base_cnn: BaseCNN = base_cnn
        self.weights: str = weights
        self.metrics: List[str] = metrics
        self.metric_accuracy: str = metric_accuracy
        self.splits_dir: Path = splits_dir
        self.min_accuracy: float = min_accuracy
        self.dataset_dir: Path = dataset_dir
        self.batch_size: int = batch_size

        # Set logs dir paths
        self.cp_dir: Path = cp_dir
        self.bu_dir: Path = sv_dir
        self.py_dir: Path = py_dir
        self.tb_dir: Path = tb_dir
        self.cm_dir: Path = cm_dir
        self.completed_marker_path = Path(self.cp_dir / "completed")

        # Set other attributes
        self.run_status: dict = {}
        self.class_names: List[str] = class_names
        # File writer for writing confusion matrix plots
        self.cm_file_writer = tf.summary.create_file_writer(str(self.tb_dir / "cm"))
        # File writer for writing evaluations against test data
        self.test_file_writer = tf.summary.create_file_writer(str(self.tb_dir / "test"))
        # Set an epoch tracker
        self.epochs_completed = 0
        self.accuracy_best = 0.0

    def is_completed(self) -> bool:
        return self.completed_marker_path.exists()

    def execute(self) -> float:
        # Create log dirs
        self.cp_dir.mkdir(parents=True, exist_ok=True)
        self.py_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.cm_dir.mkdir(parents=True, exist_ok=True)

        # Set strategy
        strategy = tf.distribute.MirroredStrategy()

        # Build datasets
        self.train_ds = self._get_dataset_split("train")
        self.val_ds = self._get_dataset_split("val")
        self.test_ds = self._get_dataset_split("test")

        with strategy.scope():
            # Create model
            self.model = tf.keras.models.Sequential(
                name=self.name,
                layers=[
                    tf.keras.layers.experimental.preprocessing.RandomFlip(
                        "horizontal", input_shape=(*self.base_cnn.image_size, 3)
                    ),
                    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
                    self.base_cnn.base_model(
                        include_top=False,
                        weights=None
                        if self.weights.lower() == "none"
                        else self.weights,
                        classes=len(self.class_names),
                    ),
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dropout(0.1),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(512, activation="relu"),
                    tf.keras.layers.Dense(
                        len(self.class_names), activation="softmax", name="predictions"
                    ),
                ],
            )
            # Compile model
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=self.metrics,
            )

        # Log model summary
        self.model.summary(line_length=80, print_fn=log.info)

        # Train
        self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.max_epochs,
            use_multiprocessing=True,
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    log_dir=self.tb_dir,
                    histogram_freq=0,
                    update_freq="epoch",
                    write_graph=True,
                    write_images=True,
                    write_steps_per_second=True,
                    profile_batch=0,
                ),
                tf.keras.callbacks.LambdaCallback(
                    on_epoch_begin=self._on_epoch_start,
                    on_epoch_end=self._on_epoch_end,
                ),
                tf.keras.callbacks.experimental.BackupAndRestore(self.cp_dir),
            ],
        )

        if self.epochs_completed >= self.max_epochs:
            self.run_status.update(
                {"reason": "max_epochs", "epoch": self.epochs_completed}
            )
            with open(self.completed_marker_path, "w") as f:
                f.write(json.dumps(self.run_status))
        else:
            self.run_status.update(
                {"reason": "early_stopped", "epoch": self.epochs_completed}
            )

        with open(self.completed_marker_path, "w") as f:
            f.write(json.dumps(self.run_status))
        log.info(f"Training run done: {self.run_status['reason']}")

        # Return best accuracy value for hparams metric
        return self.accuracy_best

    def _get_dataset_split(self, split: str) -> tf.data.Dataset:
        _options = tf.data.Options()
        _options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )
        _options.threading.max_intra_op_parallelism = 1

        def _preprocess(img, lbl):
            return self.base_cnn.preprocess(img), lbl

        return (
            tf.keras.preprocessing.image_dataset_from_directory(
                self.splits_dir / split,
                labels="inferred",
                label_mode="int",
                image_size=self.base_cnn.image_size,
                batch_size=self.batch_size,
                shuffle=True,
                seed=SEED,
            )
            .with_options(_options)
            .map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            .shuffle(
                buffer_size=1024,
                seed=SEED,
                reshuffle_each_iteration=True,
            )
        )

    def _log_confusion_matrix(self, epoch, logs=None):
        pred_y, true_y = [], []
        for batch_X, batch_y in self.test_ds:
            true_y.extend(batch_y)
            pred_y.extend(np.argmax(self.model.predict(batch_X), axis=-1))
        cm_data = sklearn.metrics.confusion_matrix(true_y, pred_y)
        cm_figure = plot_confusion_matrix(
            np.nan_to_num(cm_data),
            class_names=[
                os.path.basename(i).replace("architecture", "").replace("style", "")
                for i in self.class_names
            ],
        )
        cm_image = plot_to_image(cm_figure, file_name=self.cm_dir / f"{epoch}.png")
        with self.cm_file_writer.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)

    def _on_epoch_start(self, epoch, logs=None):
        log.info(f"Training {self.name}...")

    def _on_epoch_end(self, epoch, logs=None):
        self.epochs_completed = epoch

        # Report epoch end
        log.info(f"Epoch {epoch} done: {logs}")

        # Log confusion matrix
        self._log_confusion_matrix(epoch, logs)

        test_loss, test_accuracy = self._evaluate_against_test_data(
            epoch, write_to_tensorboard=True
        )

        if test_accuracy >= self.accuracy_best:
            self.accuracy_best = test_accuracy
            if test_accuracy >= self.min_accuracy:
                self._save_model(epoch, test_loss, test_accuracy)

    def _evaluate_against_test_data(
        self, epoch: int, write_to_tensorboard: bool = False
    ):
        log.info("Evaluating against test data...")
        test_loss, test_accuracy = self.model.evaluate(self.test_ds)
        results = {
            "epoch": epoch,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }
        log.info(f"Evaluation against test data done: {results}")
        if write_to_tensorboard:
            with self.test_file_writer.as_default():
                tf.summary.scalar("test_loss", test_loss, step=epoch)
                tf.summary.scalar("test_accuracy", test_accuracy, step=epoch)
        return test_loss, test_accuracy

    def _save_model(self, epoch, test_loss, test_accuracy):
        log.info("Saving model...")
        path = self.bu_dir / f"{self.name}-{epoch}-{test_loss:.4f}-{test_accuracy:.4f}"
        self.model.save(
            filepath=path,
            overwrite=True,
            include_optimizer=True,
            save_format="tf",
            signatures=None,
            options=tf.saved_model.SaveOptions(
                save_debug_info=True, experimental_variable_policy=None
            ),
            save_traces=True,
        )
        log.info(f"Model saved: {path.name}")
