import json
import logging
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import settings
import sklearn
import sklearn.metrics
import tensorflow as tf
from models import CNN_APPS
from plotting import plot_confusion_matrix, plot_to_image
from settings import CP_DIR, DATASET_DIR, PY_LOGS_DIR, SEED, TB_LOGS_DIR

log = logging.getLogger(settings.APP_NAME)


class TrainingRun:
    def __init__(
        self,
        name: str,
        max_epochs: int,
        profile: bool,
        backup_freq: int,
        test_freq: int,
        patience: float,
        cnn_model: tf.keras.Model,
        weights: str,
        learning_rate: float,
        metrics: List[str],
        splits_dir: Path,
    ):
        self.name: str = name
        self.max_epochs: int = max_epochs
        self.profile: bool = profile
        self.backup_freq: int = backup_freq
        self.test_freq: int = test_freq
        self.patience: float = patience
        self.cnn_model: tf.keras.Model = cnn_model
        self.weights: str = weights
        self.learning_rate: float = learning_rate
        self.metrics: List[str] = metrics
        self.splits_dir: Path = splits_dir

        cnn_app = CNN_APPS[self.cnn_model]
        self.cnn_app_model: Callable = cnn_app["class"]
        self.cnn_app_preprocessor: Callable = cnn_app["preprocessor"]
        self.image_size: Tuple[int] = cnn_app["image_size"]
        self.batch_size: int = cnn_app["batch_size"]

    def execute(self):
        class_names = list([i.name for i in DATASET_DIR.iterdir()])
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.RandomFlip(
                    "horizontal", input_shape=(*self.image_size, 3)
                ),
                tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
                self.cnn_app_model_class(
                    include_top=False,
                    weights=self.weights or None,
                    classes=len(class_names),
                ),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation="relu"),
                # tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(
                    len(class_names),
                    activation="softmax",
                    name="predictions",
                ),
            ]
        )

        # latest_epoch = 0
        completed_file_path = Path(PY_LOGS_DIR / self.name / "completed")
        run_status: dict = {}
        if completed_file_path.exists():
            with open(completed_file_path, "r") as f:
                run_status = json.loads(f.read())
        else:
            with open(completed_file_path, "w") as f:
                f.write(json.dumps(run_status))

        # Log model summary
        model.summary(print_fn=log.info)

        # Compile
        adam = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=adam,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=self.metrics,
        )

        train_ds = self.get_dataset("train")
        val_ds = self.get_dataset("val")
        test_ds = self.get_dataset("test")

        # Define a file writer for confusion matrix logging
        cm_file_writer = tf.summary.create_file_writer(
            str(TB_LOGS_DIR / self.name / "cm")
        )
        test_file_writer = tf.summary.create_file_writer(
            str(TB_LOGS_DIR / self.name / "test")
        )

        def log_confusion_matrix(epoch, logs):
            pred_y, true_y = [], []
            for batch_X, batch_y in test_ds:
                true_y.extend(batch_y)
                pred_y.extend(np.argmax(model.predict(batch_X), axis=-1))
            cm_data = np.nan_to_num(sklearn.metrics.confusion_matrix(true_y, pred_y))
            cm_figure = plot_confusion_matrix(cm_data, class_names=class_names)
            cm_image = plot_to_image(cm_figure)
            with cm_file_writer.as_default():
                tf.summary.image("Confusion Matrix", cm_image, step=epoch)

        cp_run_dir = self.get_checkpoints_run_dir(self.name)

        def on_epoch_end(epoch, logs):

            # Report epoch end
            log.info(f"Epoch {epoch} done", logs)

            # Log confusion matrix
            log_confusion_matrix(epoch, logs)

            # Set task booleans
            do_test = (
                self.test_freq and epoch > 0 and epoch % self.test_freq == 0
            ) or epoch == self.max_epochs
            do_backup = (
                self.backup_freq and epoch > 0 and epoch % self.backup_freq == 0
            ) or epoch == self.max_epochs

            # Test
            if do_test or do_backup:
                log.info("Evaluating model against test data...")
                test_loss, test_accuracy = model.evaluate(test_ds)
                results = {
                    "epoch": epoch,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                }
                log.info(f"Model evaluation results: {results}")
                if not do_backup:
                    with test_file_writer.as_default():
                        tf.summary.scalar("test_loss", test_loss, step=epoch)
                        tf.summary.scalar("test_accuracy", test_accuracy, step=epoch)
                model_backup_path = (
                    cp_run_dir
                    / f"{self.name}-{epoch}-{test_loss:.4f}-{test_accuracy:.4f}"
                )

            # Backup
            if do_backup:
                log.info("Saving model backup...")
                model.save(
                    filepath=model_backup_path,
                    overwrite=True,
                    include_optimizer=True,
                    save_format="tf",
                    signatures=None,
                    options=tf.saved_model.SaveOptions(
                        save_debug_info=True, experimental_variable_policy=None
                    ),
                    save_traces=True,
                )
                log.info(f"Model backup saved: {model_backup_path.name}")

            # Set completed file marker if max_epochs is reached
            if epoch >= self.max_epochs:
                reason = f"max epochs reached ({epoch})"
                run_status.update({"reason": reason})
                with open(completed_file_path, "w") as f:
                    f.write(json.dumps(run_status))
                log.info(f"Training run complete: {reason}")

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.max_epochs,
            use_multiprocessing=True,
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    log_dir=TB_LOGS_DIR / self.name,
                    histogram_freq=0,
                    update_freq="epoch",
                    write_graph=True,
                    write_images=True,
                    write_steps_per_second=True,
                    profile_batch=(2, 8) if self.profile else 0,
                ),
                tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end),
                tf.keras.callbacks.experimental.BackupAndRestore(cp_run_dir),
                tf.keras.callbacks.EarlyStopping(
                    min_delta=0.0001,
                    patience=self.patience,
                    restore_best_weights=True,
                ),
            ],
        )

        if not completed_file_path.exists():
            with open(completed_file_path, "w") as f:
                f.write(json.dumps({"reason": f"early stopped {self.patience}"}))

        _, test_accuracy = model.evaluate(test_ds)
        return test_accuracy

    def get_checkpoints_run_dir(self, run_name) -> Path:
        return CP_DIR / run_name

    def get_dataset(self, split: str) -> tf.data.Dataset:
        _options = tf.data.Options()
        _options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )
        _options.threading.max_intra_op_parallelism = 1

        def _preprocess(img, lbl):
            return self.cnn_app_preprocessor(img), lbl

        return (
            tf.keras.preprocessing.image_dataset_from_directory(
                self.splits_dir / split,
                labels="inferred",
                label_mode="int",
                image_size=self.image_size,
                batch_size=self.batch_size,
                shuffle=True,
                seed=SEED,
            )
            .with_options(_options)
            .map(_preprocess, num_parallel_calls=16)
            .cache()
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            .shuffle(
                buffer_size=1024,
                seed=SEED,
                reshuffle_each_iteration=True,
            )
        )
