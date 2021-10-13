import operator
import os
from functools import reduce
from pathlib import Path
from typing import Optional

import numpy as np
import sklearn
import sklearn.metrics
import tensorboard
import tensorflow as tf
import tensorflow_datasets as tfds
from cnn import CNN_APPS
from tensorboard.plugins.hparams import api as hp
from utils import generate_splits, plot_confusion_matrix, plot_to_image

PREPROCESS_SEED = 123456

MODULE_BASE_DIR = Path(__file__).parent.absolute()
REPO_DIR = MODULE_BASE_DIR.parent
OUTPUT_DIR = REPO_DIR / "output"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
LOGS_DIR = OUTPUT_DIR / "logs"


# Generate Dataset Splits
DATASET_SOURCE_DIR = Path(REPO_DIR / "dataset")
if not DATASET_SOURCE_DIR.exists() or not list(DATASET_SOURCE_DIR.iterdir()):
    raise EnvironmentError(
        "If running module directly, add source dataset to ./dataset "
        "with structure root/classes/images"
    )
DATASET_TOTAL_COUNT = reduce(
    operator.add,
    (len(list(d.iterdir())) for d in DATASET_SOURCE_DIR.iterdir()),
)
DATASET_SPLIT_DIR = generate_splits(
    src_dir=DATASET_SOURCE_DIR,
    dst_dir=Path(REPO_DIR, "input").absolute(),
    seed=PREPROCESS_SEED,
)
DATASET_BUILDER = tfds.ImageFolder(DATASET_SPLIT_DIR)
CLASS_NAMES = [
    DATASET_BUILDER.info.features["label"].int2str(n)
    for n in range(DATASET_BUILDER.info.features["label"].num_classes)
]


# Prepare hyperparameters
HP_CNN_MODEL = hp.HParam("model", hp.Discrete(list(CNN_APPS.keys())))
HP_WEIGHTS = hp.HParam("weights", hp.Discrete(["", "imagenet"]))
HP_POOLING = hp.HParam("weights", hp.Discrete(["avg", "max"]))
HP_LEARNING_RATE = hp.HParam(
    "learning_rate",
    # hp.Discrete([float(1e-4), float(3e-4), float(5e-4)]),
    hp.Discrete([float(1e-4), float(5e-4)]),
)

METRIC_ACCURACY = "accuracy"

with tf.summary.create_file_writer(str(LOGS_DIR / "hparam_tuning")).as_default():
    hp.hparams_config(
        hparams=[HP_CNN_MODEL, HP_WEIGHTS, HP_LEARNING_RATE],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name="Test Accuracy")],
    )

# Define training run function
def execute_run(hparams, run_name):
    cnn_app = CNN_APPS[hparams[HP_CNN_MODEL]]

    def get_dataset(split):
        _options = tf.data.Options()
        _options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )
        _options.threading.max_intra_op_parallelism = 1
        return (
            DATASET_BUILDER.as_dataset(split=split, batch_size=cnn_app["batch_size"])
            .with_options(_options)
            .map(
                lambda tensor: (
                    tf.image.resize(tensor["image"], cnn_app["image_size"]),
                    tensor["label"],
                )
            )
            .map(
                lambda img, _: (cnn_app["preprocessor"](img), _), num_parallel_calls=16
            )
            .cache()
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            .shuffle(
                buffer_size=DATASET_TOTAL_COUNT,
                seed=PREPROCESS_SEED,
                reshuffle_each_iteration=True,
            )
        )

    train_ds = get_dataset("train")
    val_ds = get_dataset("val")
    test_ds = get_dataset("test")

    def restore_weights_from_checkpoint(model):
        if not CHECKPOINTS_DIR.exists():
            return model
        latest_cp = tf.train.latest_checkpoint(CHECKPOINTS_DIR)
        if latest_cp:
            model.load_weights(latest_cp)
            _, restored_test_acc = model.evaluate(test_ds, verbose=2)
            print(f"Restored model test accuracy: {restored_test_acc}")
        return model

    # def get_cnn_model(hparams):
    #     # kwargs = dict(
    #     #     include_top=True,
    #     #     weights=hparams[HP_WEIGHTS] if hparams[HP_WEIGHTS] else None,
    #     # )
    #     if hparams[HP_WEIGHTS] != "imagenet":
    #         kwargs["classes"] = len(CLASS_NAMES)
    #     return CNN_APPS[hparams[HP_CNN_MODEL]]["model"](**kwargs)

    def get_fine_tune_cnn_model(
        classes: int, weights: Optional[str] = None, pooling: str = "avg"
    ):
        return tf.keras.models.Sequential(
            [
                tf.keras.applications.InceptionResNetV2(
                    include_top=False,
                    weights=weights if weights else None,
                    input_tensor=None,
                    input_shape=None,
                    pooling=None,
                    classes=classes,
                ),
                tf.keras.layers.GlobalAveragePooling2D()
                if pooling == "avg"
                else tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation="relu"),
                # tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(
                    classes, activation="softmax", name="predictions"
                ),
            ]
        )

    gpus = tf.config.list_logical_devices("GPU")
    with tf.distribute.MirroredStrategy(gpus).scope():

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
                    get_fine_tune_cnn_model(
                        classes=len(CLASS_NAMES), weights=hparams[HP_WEIGHTS]
                    ),
                ]
            )
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hparams[HP_LEARNING_RATE]),
            loss="sparse_categorical_crossentropy",
            metrics=[METRIC_ACCURACY],
        )

    # Defining a file writer for confusion matrix logging
    cm_file_writer = tf.summary.create_file_writer(str(LOGS_DIR / run_name / "cm"))

    def log_confusion_matrix(epoch, logs):
        pred_y, true_y = [], []
        for batch_X, batch_y in test_ds:
            true_y.extend(batch_y)
            pred_y.extend(np.argmax(model.predict(batch_X), axis=-1))
        cm_data = np.nan_to_num(sklearn.metrics.confusion_matrix(true_y, pred_y))
        cm_figure = plot_confusion_matrix(cm_data, class_names=CLASS_NAMES)
        cm_image = plot_to_image(cm_figure)
        with cm_file_writer.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=LOGS_DIR / run_name,
                histogram_freq=1,
                profile_batch=0,
                # profile_batch=(10, 20),
                # write_graph=True,
                # write_images=True,
            ),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(CHECKPOINTS_DIR)
                + f"/{run_name}"
                + "-epoch-{epoch:04d}.ckpt",
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


def launch_tensorboard(logs_dir):
    tb = tensorboard.program.TensorBoard()
    tb.configure(argv=[None, f"--logdir={logs_dir}", "--bind_all"])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")


def train_all():
    run_num = 0
    for cnn_model in HP_CNN_MODEL.domain.values:
        for weights in HP_WEIGHTS.domain.values:
            for learning_rate in HP_LEARNING_RATE.domain.values:
                hparams = {
                    HP_CNN_MODEL: cnn_model,
                    HP_WEIGHTS: weights,
                    HP_LEARNING_RATE: learning_rate,
                }
                run_name = (
                    f"run-{run_num}"
                    f"-{cnn_model}"
                    f"-{weights if weights else 'none'}"
                    f"-{learning_rate}"
                )
                run_logs_dir = LOGS_DIR / run_name
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
                            hp.hparams(hparams)  # record the values used in this run
                            test_accuracy = execute_run(hparams, run_name)
                            tf.summary.scalar(METRIC_ACCURACY, test_accuracy, step=1)
                    except Exception as err:
                        raise err
                    else:
                        with open(run_completed_file_path, "w") as f:
                            f.write(f"Test Accuracy: {test_accuracy}")
                run_num += 1


def main():

    launch_tensorboard(LOGS_DIR)
    train_all()


if __name__ == "__main__":
    main()
