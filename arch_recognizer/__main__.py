from __future__ import absolute_import

import sys
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent.absolute()


def main():
    from . import cli

    parser = cli.get_parser()
    args = parser.parse_args(sys.argv[1:])
    if getattr(args, "func", None) is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
