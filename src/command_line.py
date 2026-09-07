"""Dispatch the repository's explicitly listed workflow commands."""
import argparse
from importlib import import_module
import sys


def dispatch(commands: dict[str, str], description: str, argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("workflow", choices=tuple(commands))
    parser.epilog = "Use WORKFLOW --help for its inputs and options."
    selected = parser.parse_args(arguments[:1])
    return import_module(commands[selected.workflow]).main(arguments[1:])
