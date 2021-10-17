import argparse

from . import commands


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

    # create subparsers
    subparsers = parser.add_subparsers(help="sub-command help")

    # train command
    parser_train = subparsers.add_parser("train", help="train model")
    parser_train.set_defaults(func=commands.train)
    parser_train.add_argument(
        "-p",
        "--patience",
        default=20,
        type=float,
        help="early stopping patience (default: %(default)s)",
    )
    parser_train.add_argument(
        "-b",
        "--backup-freq",
        default=0,
        type=int,
        help="frequency of model backups in number of epochs (model will first be "
        "evaluated against test data and results will be included in the file name but "
        "not sent to Tensorboard) (default: %(default)s)",
    )
    parser_train.add_argument(
        "-t",
        "--test-freq",
        default=5,
        type=int,
        help="frequency to evaluate model against test data in number of epochs "
        "(default: %(default)s)",
    )
    parser_train.add_argument(
        "-e",
        "--max-epochs",
        default=300,
        type=int,
        help="maximum epochs per run (default: %(default)s)",
    )
    parser_train.add_argument(
        "-d",
        "--data-proportion",
        default=1.0,
        type=float,
        help="proportion of dataset to use (default: %(default)s)",
    )
    parser_train.add_argument(
        "--profile",
        action="store_true",
        help="enable performance profiling (default: %(default)s)",
    )
    parser_train.add_argument(
        "--eager",
        action="store_true",
        help="enable eager execution of tf.function calls (default: %(default)s)",
    )

    # test command
    parser_test = subparsers.add_parser(
        "test", help="evaluate a trained model with test data"
    )
    parser_test.set_defaults(func=commands.test)
    parser_test.add_argument(
        "run_number",
        type=int,
        help="run number of the model file/directory",
    )

    return parser
