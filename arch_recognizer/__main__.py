import argparse
import sys
from pathlib import Path

import cnn
import tensorflow as tf
import training

REPO_DIR = Path(__file__).parent.parent.absolute()


def test(args):
    trainer = training.Trainer()

    def _get_run_checkpoints_dir():
        for f in training.CHECKPOINTS_DIR.iterdir():
            if f.name.startswith("run-"):
                n = int(f.name.replace("run-", "")[0])
            else:
                n = int(f.name[0])
            if n == args.run_number:
                return Path(f)

    run_checkpoints_dir = _get_run_checkpoints_dir()
    if not run_checkpoints_dir:
        print(f"run '{args.run_number}' not found")
    else:
        latest_checkpoint_path = tf.train.latest_checkpoint(run_checkpoints_dir)
        model = tf.keras.models.load_model(f"{latest_checkpoint_path}.ckpt")
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
    parser_test = subparsers.add_parser(
        "test", help="evaluate a trained model with test data"
    )
    parser_test.set_defaults(func=test)
    parser_test.add_argument(
        "run_number",
        type=int,
        help="Run number of the model file/directory",
    )

    args = parser.parse_args(sys.argv[1:])
    if getattr(args, "func", None) is None:
        parser.print_help()
    else:
        args.func(args)
