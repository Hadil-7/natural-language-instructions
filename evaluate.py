"""Compatibility entry point for `python evaluate.py [options]`."""

import sys

from main import main


if __name__ == "__main__":
    raise SystemExit(main(["evaluate", *sys.argv[1:]]))
