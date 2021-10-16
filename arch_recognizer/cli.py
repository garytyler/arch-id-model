import argparse

import commands


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
        "-e",
        "--max-epochs",
        default=100,
        type=int,
        help="maximum epochs per run (default: %(default)s)",
    )
    parser_train.add_argument(
        "-p",
        "--dataset-proportion",
        default=1.0,
        type=float,
        help="proportion of dataset (default: %(default)s)",
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
