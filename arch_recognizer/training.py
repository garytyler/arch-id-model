import logging
import operator
import os
import tempfile
from functools import reduce
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import sklearn
import sklearn.metrics
import tensorboard
import tensorflow as tf
import tensorflow_datasets as tfds
from cnn import CNN_APPS
from plotting import plot_confusion_matrix, plot_to_image
from splitting import generate_dataset_splits
from tensorboard.plugins.hparams import api as hp

BASE_DIR: Path = Path(__file__).parent.parent.absolute()
SOURCE_DIR: Path = BASE_DIR / "dataset"
OUTPUT_DIR: Path = BASE_DIR / "output"

log = logging.getLogger(__name__)


class Trainer:
    seed = 123456

    def __init__(self):
        self.source_dir = SOURCE_DIR
        self.output_dir = OUTPUT_DIR
        self.checkpoints_dir: Path = self.output_dir / "checkpoints"
        self.logs_dir: Path = self.output_dir / "logs"
        self.input_dir: Path = BASE_DIR / "input"
        # self.input_dir: tempfile.gettempdir()

        if not self.source_dir.exists() or not list(self.source_dir.iterdir()):
            raise EnvironmentError(
                "If running module directly, add source dataset to ./dataset "
                "with structure root/classes/images"
            )
        self.dataset_total_count = reduce(
            operator.add,
            (len(list(d.iterdir())) for d in self.source_dir.iterdir()),
        )

        os.makedirs(self.input_dir, exist_ok=True)
        self.dataset_builder = tfds.ImageFolder(self.input_dir)
        self.class_names = [
            self.dataset_builder.info.features["label"].int2str(n)
            for n in range(self.dataset_builder.info.features["label"].num_classes)
        ]

    def get_training_runs(self):
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
        return (
            # self.dataset_builder.as_dataset(split=split, batch_size=batch_size)
            tf.keras.preprocessing.image_dataset_from_directory(
                self.input_dir / split,
                labels="inferred",
                label_mode="int",
                image_size=image_size,
                batch_size=batch_size,
                shuffle=True,
                seed=self.seed,
            )
            .with_options(_options)
            .map(lambda img, lbl: (preprocessor(img), lbl), num_parallel_calls=16)
            .cache()
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            # .shuffle(
            #     buffer_size=self.dataset_total_count,
            #     seed=self.seed,
            #     # reshuffle_each_iteration=True,
            # )
        )

    # Define training run function
    def execute_run(self, hparams, run_name: str, total_epochs: int) -> Optional[float]:
        # Skip or restore checkpoint

        latest_epoch = 0
        checkpoints_run_dir = self.checkpoints_dir / run_name
        latest_checkpoint_path = tf.train.latest_checkpoint(checkpoints_run_dir)
        early_stopped_file_path = Path(self.logs_dir / run_name / "early_stopped")
        if not latest_checkpoint_path:
            print(f"Starting: {run_name}")
        else:
            latest_epoch = int(
                Path(latest_checkpoint_path).name.replace(f"{run_name}-", "")[:4]
            )
            if latest_epoch >= total_epochs:
                print(f"Skipping: {run_name} ({total_epochs} completed)")
                return None
            elif early_stopped_file_path.exists():
                with open(early_stopped_file_path, "r") as f:
                    print(f"Skipping: {run_name} (early stopped: {f.read()})")
                return None
            else:
                print(f"Restored: {Path(latest_checkpoint_path).name}")

        cnn_app = CNN_APPS[hparams[self.hp_cnn_model]]

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

        # class Net:
        #     def __init__(self):
        #         super(self).__init__()

        #         tf.keras.layers.experimental.preprocessing.RandomFlip(
        #             "horizontal", input_shape=(*cnn_app["image_size"], 3)
        #         )(x)
        #         tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),

        model = tf.keras.models.Sequential(
            [
                # Preprocessing
                # tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255),
                # cnn_app["preprocessor"],
                # Augmentation
                tf.keras.layers.experimental.preprocessing.RandomFlip(
                    "horizontal", input_shape=(*cnn_app["image_size"], 3)
                ),
                tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
                # Convolution
                # get_cnn_model(hparams),
                # self.get_cnn_model(weights=hparams[self.hp_weights]),
                cnn_app["class"](
                    include_top=True,
                    weights=None,
                    classes=len(self.class_names),
                ),
                # tf.keras.layers.GlobalAveragePooling2D()
                # # if pooling == "avg"
                # # else tf.keras.layers.GlobalAveragePooling2D(),
                # ,
                # tf.keras.layers.Dropout(0.1),
                # tf.keras.layers.Flatten(),
                # tf.keras.layers.Dense(256, activation="relu"),
                # # tf.keras.layers.Dropout(0.1),
                # tf.keras.layers.Dense(
                #     len(self.class_names), activation="softmax", name="predictions"
                # ),
            ]
        )
        # model = cnn_app["class"](
        #     include_top=True,
        #     weights=None,
        #     classes=len(self.class_names),
        # )

        # Compile
        adam = tf.keras.optimizers.Adam(learning_rate=hparams[self.hp_learning_rate])
        model.compile(
            optimizer=adam,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[self.metric_accuracy],
        )

        # Defining a file writer for confusion matrix logging
        cm_file_writer = tf.summary.create_file_writer(
            str(self.logs_dir / run_name / "cm")
        )

        def log_confusion_matrix(epoch, logs):
            pred_y, true_y = [], []
            for batch_X, batch_y in test_ds:
                true_y.extend(batch_y)
                pred_y.extend(np.argmax(model.predict(batch_X), axis=-1))
            cm_data = np.nan_to_num(sklearn.metrics.confusion_matrix(true_y, pred_y))
            cm_figure = plot_confusion_matrix(cm_data, class_names=self.class_names)
            cm_image = plot_to_image(cm_figure)
            with cm_file_writer.as_default():
                tf.summary.image("Confusion Matrix", cm_image, step=epoch)

        checkpoint_basename = (
            f"{run_name}" + "-{epoch:04d}" + "-{val_loss:.2f}" + "-{val_accuracy:.2f}"
        )

        early_stopping_cb = tf.keras.callbacks.EarlyStopping(
            # monitor="val_accuracy",
            min_delta=0.0001,
            patience=20,
            restore_best_weights=True,
        )

        print(early_stopping_cb.patience)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=total_epochs,
            use_multiprocessing=True,
            initial_epoch=latest_epoch,
            callbacks=[
                early_stopping_cb,
                tf.keras.callbacks.TensorBoard(
                    log_dir=self.logs_dir / run_name,
                    histogram_freq=1,
                    profile_batch=0,
                    # profile_batch=(10, 20),
                    # write_graph=True,
                    # write_images=True,
                ),
                tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix),
                # Save model
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=f"{checkpoints_run_dir}/{checkpoint_basename}.ckpt",
                    verbose=1,
                    save_best_only=True,
                    save_freq="epoch",
                ),
                # Save weights
                tf.keras.callbacks.ModelCheckpoint(
                    # monitor="val_accuracy",
                    # filepath=.hd5",
                    filepath=f"{checkpoints_run_dir}/{checkpoint_basename}",
                    verbose=1,
                    save_weights_only=True,
                    save_best_only=True,
                    save_freq="epoch",
                ),
            ],
        )

        with open(early_stopped_file_path, "w") as f:
            f.write(str(early_stopping_cb.patience) + "\n\n" + str(history))

        print(history)

        _, accuracy = model.evaluate(test_ds)
        return accuracy

    def launch_tensorboard(self):
        tb = tensorboard.program.TensorBoard()
        tb.configure(argv=[None, f"--logdir={self.logs_dir}", "--bind_all"])
        url = tb.launch()
        print(f"Tensorflow listening on {url}")

    def train(self, dataset_proportion: float, epochs: int):
        generate_dataset_splits(
            src_dir=self.source_dir,
            dst_dir=self.input_dir,
            seed=self.seed,
            proportion=dataset_proportion,
        )

        training_runs = self.get_training_runs()

        with tf.summary.create_file_writer(
            str(self.logs_dir / "hparam_tuning")
        ).as_default():
            hp.hparams_config(
                hparams=[self.hp_cnn_model, self.hp_weights, self.hp_learning_rate],
                metrics=[hp.Metric(self.metric_accuracy, display_name="Accuracy")],
            )

        for run_num, hparams in enumerate(training_runs):
            run_name = (
                f"run-{run_num}"
                f"-{hparams[self.hp_cnn_model]}"
                f"-{hparams[self.hp_weights] or 'none'}"
                f"-{hparams[self.hp_learning_rate]}"
            )
            run_logs_dir = self.logs_dir / run_name
            run_logs_file_writer = tf.summary.create_file_writer(
                logdir=str(run_logs_dir)
            )
            run_completed_file_path = Path(run_logs_dir / "completed")

            # else:
            #     print(f"Starting run: {run_name}")
            #     try:
            #         with run_logs_file_writer.as_default():
            #             hp.hparams(hparams)
            #             with tf.distribute.MirroredStrategy().scope():
            #                 test_accuracy = self.execute_run(
            #                     hparams, run_name, epochs=epochs
            #                 )
            #                 if test_accuracy is None:
            #                     print(f"Skipping run: {run_name} ({f.read()})")
            #             tf.summary.scalar(self.metric_accuracy, test_accuracy, step=1)
            #     except Exception as err:
            #         raise err
            #     else:
            #         with open(run_completed_file_path, "w") as f:
            #             f.write(f"Test Accuracy: {test_accuracy}")

            with run_logs_file_writer.as_default():
                hp.hparams(hparams)
                with tf.distribute.MirroredStrategy().scope():
                    accuracy = self.execute_run(hparams, run_name, total_epochs=epochs)
                    if accuracy is None:
                        continue

                    tf.summary.scalar(self.metric_accuracy, accuracy, step=1)
            # except Exception as err:
            #     raise err
            # else:
            #     with open(run_completed_file_path, "w") as f:
            #         f.write(f"Test Accuracy: {test_accuracy}")
