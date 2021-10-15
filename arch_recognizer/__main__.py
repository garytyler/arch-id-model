import argparse
import sys
from pathlib import Path

import cnn
import numpy as np
import sklearn
import tensorflow as tf
import training

REPO_DIR = Path(__file__).parent.parent.absolute()


def test(args):
    trainer = training.Trainer()
    checkpoints_run_dir = trainer.get_checkpoints_run_dir(args.run_name)
    latest_checkpoint_path = tf.train.latest_checkpoint(checkpoints_run_dir)
    model = tf.keras.models.load_model(latest_checkpoint_path + ".ckpt")
    cnn_app = cnn.CNN_APPS["InceptionResNetV2"]
    test_ds = trainer.get_dataset(
        split="test",
        image_size=cnn_app["image_size"],
        batch_size=cnn_app["batch_size"],
        preprocessor=cnn_app["preprocessor"],
    )
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test loss: {loss}\nTest accuracy: {accuracy}")

    pred_y, true_y = [], []
    for batch_X, batch_y in test_ds:
        true_y.extend(batch_y)
        pred_y.extend(np.argmax(model.predict(batch_X), axis=-1))
    # cm_data = np.nan_to_num(sklearn.metrics.confusion_matrix(true_y, pred_y))

    print(sklearn.metrics.mean_absolute_percentage_error(true_y, pred_y))


def train(args):
    trainer = training.Trainer()
    trainer.train(
        dataset_proportion=args.dataset_proportion,
        max_epochs=args.max_epochs,
    )


if __name__ == "__main__":
    # top-level parser
    parser = argparse.ArgumentParser(prog="arch_recognizer")
    subparsers = parser.add_subparsers(help="sub-command help")

    # train command
    parser_train = subparsers.add_parser("train", help="train model")
    parser_train.set_defaults(func=train)
    parser_train.add_argument(
        "--max-epochs",
        default=100,
        type=int,
        help="Maximum epochs per run",
    )
    parser_train.add_argument(
        "--dataset-proportion",
        default=1.0,
        type=float,
        help="Proportion of dataset",
    )

    # test command
    parser_test = subparsers.add_parser("test", help="test model from file")
    parser_test.set_defaults(func=test)
    parser_test.add_argument(
        # "model_path",
        "run_name",
        type=Path,
        help="Path to the model file/directory",
    )

    args = parser.parse_args(sys.argv[1:])
    if getattr(args, "func", None) is None:
        parser.print_help()
    else:
        args.func(args)
