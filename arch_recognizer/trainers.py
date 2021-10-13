import operator
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
from tensorboard.plugins.hparams import api as hp
from utils import generate_dataset_splits, plot_confusion_matrix, plot_to_image


class Trainer:
    seed = 123456

    def __init__(self, repo_dir: Path):
        repo_dir = repo_dir.absolute()
        self.source_dir: Path = repo_dir / "dataset"
        self.output_dir: Path = repo_dir / "output"
        self.checkpoints_dir: Path = self.output_dir / "checkpoints"
        self.logs_dir: Path = self.output_dir / "logs"
        self.input_dir = repo_dir / "input"

        if not self.source_dir.exists() or not list(self.source_dir.iterdir()):
            raise EnvironmentError(
                "If running module directly, add source dataset to ./dataset "
                "with structure root/classes/images"
            )
        self.dataset_total_count = reduce(
            operator.add,
            (len(list(d.iterdir())) for d in self.source_dir.iterdir()),
        )

        self.dataset_builder = tfds.ImageFolder(self.input_dir)
        self.class_names = [
            self.dataset_builder.info.features["label"].int2str(n)
            for n in range(self.dataset_builder.info.features["label"].num_classes)
        ]

        # Prepare hyperparameters
        self.hp_cnn_model = hp.HParam("model", hp.Discrete(list(CNN_APPS.keys())))
        self.hp_weights = hp.HParam("weights", hp.Discrete(["", "imagenet"]))
        self.hp_pooling = hp.HParam("weights", hp.Discrete(["avg", "max"]))
        self.hp_learning_rate = hp.HParam(
            "learning_rate",
            # hp.Discrete([float(1e-4), float(3e-4), float(5e-4)]),
            hp.Discrete([float(1e-4), float(5e-4)]),
        )

        self.metric_accuracy = "accuracy"

        with tf.summary.create_file_writer(
            str(self.logs_dir / "hparam_tuning")
        ).as_default():
            hp.hparams_config(
                hparams=[self.hp_cnn_model, self.hp_weights, self.hp_learning_rate],
                metrics=[hp.Metric(self.metric_accuracy, display_name="Accuracy")],
            )

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
            self.dataset_builder.as_dataset(split=split, batch_size=batch_size)
            .with_options(_options)
            .map(
                lambda tensor: (
                    tf.image.resize(tensor["image"], image_size),
                    tensor["label"],
                )
            )
            .map(lambda img, _: (preprocessor(img), _), num_parallel_calls=16)
            .cache()
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            .shuffle(
                buffer_size=self.dataset_total_count,
                seed=self.seed,
                reshuffle_each_iteration=True,
            )
        )

    def get_fine_tune_cnn_model(
        self, weights: Optional[str] = None, pooling: str = "avg"
    ):
        return tf.keras.models.Sequential(
            [
                tf.keras.applications.InceptionResNetV2(
                    include_top=False,
                    weights=weights if weights else None,
                    input_tensor=None,
                    input_shape=None,
                    pooling=None,
                    classes=len(self.class_names),
                ),
                tf.keras.layers.GlobalAveragePooling2D()
                if pooling == "avg"
                else tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation="relu"),
                # tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(
                    len(self.class_names), activation="softmax", name="predictions"
                ),
            ]
        )

    # Define training run function
    def execute_run(self, hparams, run_name):
        cnn_app = CNN_APPS[hparams[self.hp_cnn_model]]

        train_ds = self.get_dataset(
            "train",
            image_size=cnn_app["image_size"],
            batch_size=cnn_app["batch_size"],
            preprocessor=cnn_app["preprocessor"],
        )
        val_ds = self.get_dataset(
            "val",
            image_size=cnn_app["image_size"],
            batch_size=cnn_app["batch_size"],
            preprocessor=cnn_app["preprocessor"],
        )
        test_ds = self.get_dataset(
            "test",
            image_size=cnn_app["image_size"],
            batch_size=cnn_app["batch_size"],
            preprocessor=cnn_app["preprocessor"],
        )

        def restore_weights_from_checkpoint(model):
            if not self.checkpoints_dir.exists():
                return model
            latest_cp = tf.train.latest_checkpoint(self.checkpoints_dir)
            if latest_cp:
                model.load_weights(latest_cp)
                _, restored_test_acc = model.evaluate(test_ds, verbose=2)
                print(f"Restored model test accuracy: {restored_test_acc}")
            return model

        with tf.distribute.MirroredStrategy(
            tf.config.list_logical_devices("GPU")
        ).scope():
            model = restore_weights_from_checkpoint(
                tf.keras.models.Sequential(
                    [
                        # Augmentation
                        tf.keras.layers.experimental.preprocessing.RandomFlip(
                            "horizontal", input_shape=(*cnn_app["image_size"], 3)
                        ),
                        tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
                        # Convolution
                        # get_cnn_model(hparams),
                        self.get_fine_tune_cnn_model(weights=hparams[self.hp_weights]),
                    ]
                )
            )

            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=hparams[self.hp_learning_rate]
                ),
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

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=100,
            callbacks=[
                tf.keras.callbacks.TensorBoard(
                    log_dir=self.logs_dir / run_name,
                    histogram_freq=1,
                    profile_batch=0,
                    # profile_batch=(10, 20),
                    # write_graph=True,
                    # write_images=True,
                ),
                tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(self.checkpoints_dir)
                    + f"/{run_name}"
                    + "-epoch-{epoch:04d}-{val_loss:.2f}-{val_accuracy:.2f}.ckpt",
                    verbose=1,
                    save_best_only=True,
                    save_freq="epoch",
                ),
                tf.keras.callbacks.EarlyStopping(
                    min_delta=0.0001,
                    patience=20,
                    restore_best_weights=True,
                ),
            ],
        )

        _, test_accuracy = model.evaluate(test_ds)
        return test_accuracy

    def launch_tensorboard(self):
        tb = tensorboard.program.TensorBoard()
        tb.configure(argv=[None, f"--logdir={self.logs_dir}", "--bind_all"])
        url = tb.launch()
        print(f"Tensorflow listening on {url}")

    def train(self, skip_splits_generation=False):
        if skip_splits_generation:
            self.input_dir = generate_dataset_splits(
                src_dir=self.source_dir, dst_dir=self.input_dir, seed=self.seed
            )

        run_num = 0
        for cnn_model in self.hp_cnn_model.domain.values:
            for weights in self.hp_weights.domain.values:
                for learning_rate in self.hp_learning_rate.domain.values:
                    hparams = {
                        self.hp_cnn_model: cnn_model,
                        self.hp_weights: weights,
                        self.hp_learning_rate: learning_rate,
                    }
                    run_name = (
                        f"run-{run_num}"
                        f"-{cnn_model}"
                        f"-{weights if weights else 'none'}"
                        f"-{learning_rate}"
                    )
                    run_logs_dir = self.logs_dir / run_name
                    run_logs_file_writer = tf.summary.create_file_writer(
                        logdir=str(run_logs_dir)
                    )
                    run_completed_file_path = Path(run_logs_dir / "completed")
                    if run_completed_file_path.exists():
                        with open(run_completed_file_path, "r") as f:
                            print(f"--- Skipping (completed): {run_name} ({f.read()})")
                    else:
                        print(f"--- Starting: {run_name}")
                        try:
                            with run_logs_file_writer.as_default():
                                hp.hparams(hparams)
                                test_accuracy = self.execute_run(hparams, run_name)
                                tf.summary.scalar(
                                    self.metric_accuracy, test_accuracy, step=1
                                )
                        except Exception as err:
                            raise err
                        else:
                            with open(run_completed_file_path, "w") as f:
                                f.write(f"Test Accuracy: {test_accuracy}")
                    run_num += 1
