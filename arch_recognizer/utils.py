import io
import itertools
import os
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
import numpy as np
import tensorflow as tf


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


def generate_splits(
    src_dir: Path,
    dst_dir: Path,
    seed: int,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    quantity_multiplier: float = 1.0,
) -> Path:
    rng = numpy.random.default_rng(seed=seed)

    val_dir = dst_dir.absolute() / "val"
    test_dir = dst_dir.absolute() / "test"
    train_dir = dst_dir.absolute() / "train"

    # Remove old
    shutil.rmtree(val_dir, ignore_errors=True)
    shutil.rmtree(test_dir, ignore_errors=True)
    shutil.rmtree(train_dir, ignore_errors=True)

    # Generate new
    for src_class_dir in src_dir.iterdir():
        src_class_files = list(src_class_dir.iterdir())
        src_class_num = round(len(src_class_files) * quantity_multiplier)

        # Shuffle files
        random.shuffle(src_class_files, random=rng.random)

        # Set counts
        val_num = round(src_class_num * val_ratio)
        test_num = round(src_class_num * test_ratio)
        train_num = src_class_num - val_num - test_num

        # Create lists
        val_files = []
        for n in range(val_num):
            val_files.append(src_class_files.pop().name)

        test_files = []
        for n in range(test_num):
            test_files.append(src_class_files.pop().name)

        train_files = []
        for n in range(train_num):
            train_files.append(src_class_files.pop().name)

        if not all(
            (
                set(test_files).isdisjoint(set(train_files)),
                set(test_files).isdisjoint(set(val_files)),
                set(train_files).isdisjoint(set(val_files)),
                src_class_num == len(test_files) + len(val_files) + len(train_files),
            )
        ):
            raise RuntimeError("Error generating dataset splits: wrong resulting count")

        # Copy val files
        val_class_dir = val_dir / src_class_dir.name
        os.makedirs(val_class_dir)
        for file_name in val_files:
            src = src_class_dir / file_name
            dst = val_class_dir / file_name
            shutil.copyfile(src, dst)

        # Copy test files
        test_class_dir = test_dir / src_class_dir.name
        os.makedirs(test_class_dir)
        for file_name in test_files:
            src = src_class_dir / file_name
            dst = test_class_dir / file_name
            shutil.copyfile(src, dst)

        # Copy train files
        train_class_dir = train_dir / src_class_dir.name
        os.makedirs(train_class_dir)
        for file_name in train_files:
            src = src_class_dir / file_name
            dst = train_class_dir / file_name
            shutil.copyfile(src, dst)

    return dst_dir
