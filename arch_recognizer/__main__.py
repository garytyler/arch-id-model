import sys
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent.absolute()


def main():
    import cli
    import loggers

    parser = cli.get_parser()
    args = parser.parse_args(sys.argv[1:])
    if getattr(args, "func", None) is None:
        parser.print_help()
    else:
        loggers.initialize_loggers(
            app_log_level=args.log_level,
            tf_log_level=args.tf_log_level,
        )
        args.func(args)


if __name__ == "__main__":
    main()
