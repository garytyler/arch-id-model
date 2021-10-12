import io
import itertools
import os
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import PIL
import sklearn
import sklearn.metrics
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

# Define functions to generate confusion matrix


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(len(class_names), len(class_names)))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


def get_images():
    BASE_DIR = Path(__file__).parent
    REPO_DIR = BASE_DIR.parent
    base_data_dir = Path(REPO_DIR, "input", "arch-recognizer-dataset").absolute()
    val_data_dir = base_data_dir / "val"
    test_data_dir = base_data_dir / "test"
    train_data_dir = base_data_dir / "train"

    # x = list(test_data_dir.iterdir())

    # for file_path in test_data_dir.iterdir():

    # image =
    # tf.image.resize_image_with_crop_or_pad(image, 5, 5)
    # dataset = tf.data.Dataset.from_generator(
    #     (
    #         tf.image.resize_image_with_crop_or_pad(
    #             tf.keras.preprocessing.image.load_img(file_path), 5, 5
    #         )
    #         for file_path in test_data_dir.iterdir()
    #     )
    # )
    # ds = images(list(test_data_dir.iterdir())[0])

    # img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    #     rescale=1.0 / 255, rotation_range=20
    # )
    # images, labels = next(img_gen.flow_from_directory(test_data_dir))
    # print(images.dtype, images.shape)
    # print(labels.dtype, labels.shape)

    # zeropad2 = tf.keras.layers.ZeroPadding2D()(leakyrelu)
    # print(x)


#
# tf.image.resize_image_with_crop_or_pad(a, 5, 5)


if __name__ == "__main__":
    get_images()
