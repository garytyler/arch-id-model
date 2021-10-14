import argparse
import sys
from pathlib import Path

import cnn
import tensorflow as tf
import trainers

REPO_DIR = Path(__file__).parent.parent.absolute()


def test(args):
    model = tf.keras.models.load_model(args.model_path)
    trainer = trainers.Trainer(REPO_DIR)
    cnn_app = cnn.CNN_APPS["InceptionResNetV2"]
    test_ds = trainer.get_dataset(
        split="test",
        image_size=cnn_app["image_size"],
        batch_size=cnn_app["batch_size"],
        preprocessor=cnn_app["preprocessor"],
    )
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test loss: {loss}\nTest accuracy: {accuracy}")


def train(args):
    trainer = trainers.Trainer(repo_dir=REPO_DIR)
    trainer.launch_tensorboard()
    trainer.train(skip_splits_generation=True)


if __name__ == "__main__":
    # top-level parser
    parser = argparse.ArgumentParser(prog="arch_recognizer")
    subparsers = parser.add_subparsers(help="sub-command help")

    # train command
    parser_train = subparsers.add_parser("train", help="train help")
    parser_train.set_defaults(func=train)

    # test command
    parser_test = subparsers.add_parser("test", help="test help")
    parser_test.set_defaults(func=test)
    parser_test.add_argument(
        "model_path", type=Path, help="Path to the model file/directory"
    )

    args = parser.parse_args(sys.argv[1:])
    args.func(args)
