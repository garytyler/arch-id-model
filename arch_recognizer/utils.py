import filecmp
import io
import itertools
import os
import random
import shutil
from pathlib import Path
from typing import Callable, Dict, Tuple

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


def generate_dataset_splits(
    src_dir: Path,
    dst_dir: Path,
    seed: int,
    ratios: Dict[str, float] = {"val": 0.15, "test": 0.15},
    quantity_multiplier: float = 1.0,
) -> Path:
    rng = numpy.random.default_rng(seed=seed)
    dst_files: dict = {"train": dict()}
    dst_files.update({k: {} for k in ratios})

    # Generate new
    for src_class_dir in src_dir.iterdir():
        src_class_files = list(src_class_dir.iterdir())
        src_class_count = round(len(src_class_files) * quantity_multiplier)

        # Shuffle files
        random.shuffle(src_class_files, random=rng.random)

        # Set counts
        counts = {k: round(src_class_count * v) for k, v in ratios.items()}
        counts["train"] = src_class_count - sum(counts.values())

        # Sort into lists
        for split_name in dst_files:
            dst_files[split_name][src_class_dir.name] = [
                src_class_files.pop().name for _ in range(counts[split_name])
            ]

    for split_name, classes in dst_files.items():
        for class_name, file_names in classes.items():
            dst_handled_paths = set()
            os.makedirs(dst_dir / split_name / class_name, exist_ok=True)
            # Copy files only if missing or if name and hash don't match
            for file_name in file_names:
                dst = dst_dir / split_name / class_name / file_name
                src = src_dir / class_name / file_name
                if not dst.exists():
                    shutil.copyfile(src, dst)
                elif not filecmp.cmp(dst, src):
                    os.remove(dst)
                    shutil.copyfile(src, dst)
                dst_handled_paths.add(dst)
            # Remove extra files in the destination directory
            dst_existing_paths = set(Path(dst_dir / split_name / class_name).iterdir())
            for unhandled_path in dst_existing_paths.difference(dst_handled_paths):
                os.remove(unhandled_path)

    return dst_dir
