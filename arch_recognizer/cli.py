import argparse
from pathlib import Path

from arch_recognizer.cnns import CNN_APPS

from . import commands
from .settings import BASE_DIR, WEIGHTS


def get_parser():
    # top-level parser
    parser = argparse.ArgumentParser(prog="arch_recognizer")
    parser.add_argument(
        "--log-level",
        default="info",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        help="app log level (default: %(default)s)",
    )
    parser.add_argument(
        "--tf-log-level",
        default="error",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        help="tensorflow log level (default: %(default)s)",
    )
    parser.add_argument(
        "-d",
        "--dataset-dir",
        default=BASE_DIR / "dataset",
        type=Path,
        help=(
            "dir w/ folders for each class, each containing images of that class "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=BASE_DIR / "output",
        type=Path,
        help="base directory for output from all sessions (default: %(default)s)",
    )

    # create subparsers
    subparsers = parser.add_subparsers(help="sub-command help")

    # predict command
    parser_predict = subparsers.add_parser("predict", help="predict model")
    parser_predict.set_defaults(func=commands.predict)
    parser_predict.add_argument(
        "-s",
        "--session",
        const=-1,
        nargs="?",
        type=int,
        help="resume last session or specified session (default: %(default)s)",
    )
    parser_predict.add_argument(
        "-m",
        "--cnn-model",
        default="InceptionResNetV2",
        type=str,
        choices=CNN_APPS.keys(),
        help="tensorflow log level (default: %(default)s)",
    )
    parser_predict.add_argument(
        "-w",
        "--weights",
        default="imagenet",
        type=str,
        choices=WEIGHTS,
        help="tensorflow log level (default: %(default)s)",
    )
    parser_predict.add_argument(
        "-i",
        "--count",
        default=1,
        type=int,
        help="number of test images to predict (default: %(default)s)",
    )

    # train command
    parser_train = subparsers.add_parser("train", help="train model")
    parser_train.set_defaults(func=commands.train)
    parser_train.add_argument(
        "-s",
        "--session",
        const=-1,
        nargs="?",
        type=int,
        help="resume last session or specified session (default: %(default)s)",
    )
    parser_train.add_argument(
        "-a",
        "--min-accuracy",
        default=0.6,
        type=float,
        help="min test accuracy for which to save model (default: %(default)s)",
    )
    parser_train.add_argument(
        "--proportion",
        dest="data_proportion",
        default=1.0,
        type=float,
        help="proportion of dataset to use (default: %(default)s)",
    )
    parser_train.add_argument(
        "--force-resume-session",
        action="store_true",
        help="skip comparing commit hash when using -r/--resume (default: %(default)s)",
    )
    parser_train.add_argument(
        "--disable-tensorboard-server",
        action="store_true",
        help="disable tensorboard server (default: %(default)s)",
    )

    return parser
