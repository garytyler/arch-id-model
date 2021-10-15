import argparse
import sys
from pathlib import Path

import cnn
import tensorflow as tf
import training

REPO_DIR = Path(__file__).parent.parent.absolute()


def test(args):
    model = tf.keras.models.load_model(args.model_path)
    trainer = training.Trainer()
    cnn_app = cnn.CNN_APPS["InceptionResNetV2"]
    test_ds = trainer.get_dataset(
        split="test",
        image_size=cnn_app["image_size"],
        batch_size=cnn_app["batch_size"],
        preprocessor=cnn_app["preprocessor"],
    )
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test loss: {loss}\nTest accuracy: {accuracy}")

    import numpy as np
    import sklearn

    pred_y, true_y = [], []
    for batch_X, batch_y in test_ds:
        true_y.extend(batch_y)
        pred_y.extend(np.argmax(model.predict(batch_X), axis=-1))
    # cm_data = np.nan_to_num(sklearn.metrics.confusion_matrix(true_y, pred_y))

    print(sklearn.metrics.mean_absolute_percentage_error(true_y, pred_y))


def train(args):
    trainer = training.Trainer()
    trainer.launch_tensorboard()
    trainer.train(dataset_proportion=args.proportion, epochs=args.epochs)


if __name__ == "__main__":
    # top-level parser
    parser = argparse.ArgumentParser(prog="arch_recognizer")
    subparsers = parser.add_subparsers(help="sub-command help")

    # train command
    parser_train = subparsers.add_parser("train", help="train help")
    parser_train.set_defaults(func=train)
    parser_train.add_argument(
        "-p", "--proportion", default=1.0, type=float, help="Proportion of dataset"
    )
    parser_train.add_argument(
        "-e", "--epochs", default=100, type=int, help="Number of epochs"
    )

    # test command
    parser_test = subparsers.add_parser("test", help="test help")
    parser_test.set_defaults(func=test)
    parser_test.add_argument(
        "model_path", type=Path, help="Path to the model file/directory"
    )

    args = parser.parse_args(sys.argv[1:])
    args.func(args)
