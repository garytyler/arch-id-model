import json
import logging
import operator
import os
import shutil
import tempfile
from functools import reduce
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import settings
import sklearn
import sklearn.metrics
import tensorboard
import tensorflow as tf
from loggers import app_log_formatter
from models import CNN_APPS
from plotting import plot_confusion_matrix, plot_to_image
from settings import APP_NAME, CP_DIR, DATASET_DIR, PY_LOGS_DIR, TB_LOGS_DIR
from splitting import generate_dataset_splits
from tensorboard.plugins.hparams import api as hp

log = logging.getLogger(settings.APP_NAME)


class Trainer:
    seed = 123456

    run_loggers = [log]

    def __init__(self):

        if not DATASET_DIR.exists() or not list(DATASET_DIR.iterdir()):
            raise EnvironmentError(
                "If running module directly, add source dataset to ./dataset "
                "with structure root/classes/images"
            )
        self.dataset_length = reduce(
            operator.add,
            (len(list(d.iterdir())) for d in DATASET_DIR.iterdir()),
        )
        self.splits_dir: Path = Path(tempfile.mkdtemp(prefix=f"{APP_NAME}-splits-"))

    def __del__(self):
        shutil.rmtree(self.splits_dir, exist_ok=True)

    def _set_run_log_file(self, path, level=logging.INFO):
        for logger in self.run_loggers:
            if getattr(self, "run_log_file_handler", None) and logger.hasHandlers():
                logger.removeHandler(self.run_log_file_handler)
        os.makedirs(path.parent, exist_ok=True)
        self.run_log_file_handler = logging.FileHandler(path)
        self.run_log_file_handler.setFormatter(app_log_formatter)
        self.run_log_file_handler.setLevel(level)
        for logger in self.run_loggers:
            logger.addHandler(self.run_log_file_handler)

    def _launch_tensorboard(self):
        tb = tensorboard.program.TensorBoard()
        tb.configure(argv=[None, f"--logdir={TB_LOGS_DIR}", "--bind_all"])
        url = tb.launch()
        log.info(f"Tensorflow listening on {url}")

    def _create_training_run_hyperparams(self):
        # Prepare hyperparameters
        self.hp_cnn_model = hp.HParam("model", hp.Discrete(list(CNN_APPS.keys())))
        self.hp_weights = hp.HParam("weights", hp.Discrete(["", "imagenet"]))
        # self.hp_pooling = hp.HParam("pooling", hp.Discrete(["avg", "max"]))
        self.hp_learning_rate = hp.HParam(
            "learning_rate",
            # hp.Discrete([float(1e-4), float(3e-4), float(5e-4)]),
            # hp.Discrete([float(1e-4), float(5e-4)]),
            hp.Discrete([float(1e-3), float(5e-3)]),
        )

        self.metric_accuracy = "accuracy"

        runs = []
        for cnn_model in self.hp_cnn_model.domain.values:
            for weights in self.hp_weights.domain.values:
                for learning_rate in self.hp_learning_rate.domain.values:
                    # runs.append((_cnn_model, _weights, _learning_rate))
                    runs.append(
                        {
                            self.hp_cnn_model: cnn_model,
                            self.hp_weights: weights,
                            self.hp_learning_rate: learning_rate,
                        }
                    )

        with tf.summary.create_file_writer(
            str(TB_LOGS_DIR / "hparam_tuning")
        ).as_default():
            hp.hparams_config(
                hparams=[self.hp_cnn_model, self.hp_weights, self.hp_learning_rate],
                metrics=[hp.Metric(self.metric_accuracy, display_name="Accuracy")],
            )

        return runs

    def get_dataset(
        self,
        split: str,
        image_size: Tuple[int, int],
        batch_size: int,
        preprocessor: Callable,
    ):
        _options = tf.data.Options()
        _options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )
        _options.threading.max_intra_op_parallelism = 1

        def _preprocess(img, lbl):
            return preprocessor(img), lbl

        return (
            tf.keras.preprocessing.image_dataset_from_directory(
                self.splits_dir / split,
                labels="inferred",
                label_mode="int",
                image_size=image_size,
                batch_size=batch_size,
                shuffle=True,
                seed=self.seed,
            )
            .with_options(_options)
            .map(_preprocess, num_parallel_calls=16)
            .cache()
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            .shuffle(
                buffer_size=self.dataset_length,
                # buffer_size=1024,
                seed=self.seed,
                reshuffle_each_iteration=True,
            )
        )

    def get_checkpoints_run_dir(self, run_name) -> Path:
        return CP_DIR / run_name

    # Define training run function
    def _execute_run(
        self, run_name: str, max_epochs: int, hparams, profile: bool
    ) -> Optional[float]:
        cnn_app = CNN_APPS[hparams[self.hp_cnn_model]]

        class_names = list([i.name for i in DATASET_DIR.iterdir()])
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.RandomFlip(
                    "horizontal", input_shape=(*cnn_app["image_size"], 3)
                ),
                tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
                cnn_app["class"](
                    include_top=False,
                    weights=hparams[self.hp_weights] or None,
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

        latest_epoch = 0
        checkpoints_run_dir = self.get_checkpoints_run_dir(run_name)
        # latest_checkpoint_path = tf.train.latest_checkpoint(checkpoints_run_dir)

        completed_file_path = Path(PY_LOGS_DIR / run_name / "completed")
        run_status: dict = {}
        if completed_file_path.exists():
            with open(completed_file_path, "r") as f:
                run_status = json.loads(f.read())
        else:
            with open(completed_file_path, "w") as f:
                f.write(json.dumps(run_status))

        # Log model summary
        model.summary(print_fn=log.info)

        def get_run_dataset(split: str) -> tf.data.Dataset:
            return self.get_dataset(
                split,
                image_size=cnn_app["image_size"],
                batch_size=cnn_app["batch_size"],
                preprocessor=cnn_app["preprocessor"],
            )

        train_ds = get_run_dataset("train")
        val_ds = get_run_dataset("val")
        test_ds = get_run_dataset("test")

        # Compile
        adam = tf.keras.optimizers.Adam(learning_rate=hparams[self.hp_learning_rate])
        model.compile(
            optimizer=adam,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[self.metric_accuracy],
        )

        # Define a file writer for confusion matrix logging
        cm_file_writer = tf.summary.create_file_writer(
            str(TB_LOGS_DIR / run_name / "cm")
        )
        test_file_writer = tf.summary.create_file_writer(
            str(TB_LOGS_DIR / run_name / "test")
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

        def on_epoch_end(epoch, logs):
            log_confusion_matrix(epoch, logs)
            if (epoch > 0 and epoch % 5 == 0) or epoch == max_epochs:
                test_loss, test_accuracy = model.evaluate(test_ds)
                log.info(
                    f"epoch={epoch}, "
                    f"test_loss={test_loss}, "
                    f"test_accuracy={test_accuracy}"
                )
                with test_file_writer.as_default():
                    tf.summary.scalar("test_loss", test_loss, step=epoch)
                    tf.summary.scalar("test_accuracy", test_accuracy, step=epoch)
                model.save(
                    filepath=CP_DIR
                    / run_name
                    / f"{run_name}-{epoch}-{test_loss}-{test_accuracy}",
                    overwrite=True,
                    include_optimizer=True,
                    save_format="tf",
                    signatures=None,
                    options=tf.saved_model.SaveOptions(
                        save_debug_info=True, experimental_variable_policy=None
                    ),
                    save_traces=True,
                )
            if epoch >= max_epochs:
                reason = f"max epochs reached ({epoch})"
                run_status.update({"reason": reason})
                with open(completed_file_path, "w") as f:
                    f.write(json.dumps(run_status))
                log.info(f"Completed training run: {reason}")

        patience: float = 50

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=max_epochs,
            use_multiprocessing=True,
            initial_epoch=latest_epoch,
            # steps_per_epoch=len(train_ds) // cnn_app["batch_size"],
            # validation_steps=len(val_ds) // cnn_app["batch_size"],
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    log_dir=TB_LOGS_DIR / run_name,
                    histogram_freq=0,
                    write_graph=False,
                    profile_batch=(2, 8) if profile else 0,
                ),
                tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end),
                tf.keras.callbacks.experimental.BackupAndRestore(checkpoints_run_dir),
                tf.keras.callbacks.ProgbarLogger(count_mode="samples"),
                tf.keras.callbacks.EarlyStopping(
                    min_delta=0.0001,
                    patience=patience,
                    restore_best_weights=True,
                ),
            ],
        )

        if not completed_file_path.exists():
            with open(completed_file_path, "w") as f:
                f.write(json.dumps({"reason": f"early stopped {patience}"}))

        _, test_accuracy = model.evaluate(test_ds)
        return test_accuracy

    def train(self, data_proportion: float, max_epochs: int, profile: bool):
        self._launch_tensorboard()

        training_runs = self._create_training_run_hyperparams()

        generate_dataset_splits(
            src_dir=DATASET_DIR,
            dst_dir=self.splits_dir,
            seed=self.seed,
            proportion=data_proportion,
        )

        for run_num, hparams in enumerate(training_runs):
            run_name = (
                f"run-{run_num}"
                f"-{hparams[self.hp_cnn_model]}"
                f"-{hparams[self.hp_weights] or 'none'}"
                f"-{hparams[self.hp_learning_rate]}"
            )
            self._set_run_log_file(PY_LOGS_DIR / run_name / f"{run_name}.log")
            hparams_file_writer = tf.summary.create_file_writer(str(TB_LOGS_DIR))
            with hparams_file_writer.as_default():
                hp.hparams(hparams)
            with tf.distribute.MirroredStrategy().scope():
                accuracy = self._execute_run(
                    run_name=run_name,
                    max_epochs=max_epochs,
                    hparams=hparams,
                    profile=profile,
                )
            with hparams_file_writer.as_default():
                tf.summary.scalar(self.metric_accuracy, accuracy, step=1)
