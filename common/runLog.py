"""Level-gated log. Does not read tasks or compute rates."""

import logging
import sys

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
_LOGGER_NAME = "agenteve"
_seen = set()


def useLevel(argv, default=INFO):
    level = default
    kept = []
    index = 0
    names = {"DEBUG": DEBUG, "INFO": INFO, "WARNING": WARNING, "ERROR": ERROR}
    while index < len(argv):
        item = argv[index]
        chosen = None
        if item in names:
            chosen = item
        elif item == "--level" and index + 1 < len(argv):
            index += 1
            chosen = argv[index]
        elif item.startswith("--level="):
            chosen = item.split("=", 1)[1]
        else:
            kept.append(item)
        if chosen is not None:
            if chosen not in names:
                print(f"Unknown log level {chosen}", file=sys.stderr)
                raise SystemExit(2)
            level = names[chosen]
        index += 1
    logging.basicConfig(level=level, format="%(levelname)s %(message)s", stream=sys.stderr, force=True)
    return kept


def info(message):
    logging.getLogger(_LOGGER_NAME).info(message)


def warn(message, *, once=None):
    if once is not None:
        if once in _seen:
            return
        _seen.add(once)
    logging.getLogger(_LOGGER_NAME).warning(message)


def error(message):
    logging.getLogger(_LOGGER_NAME).error(message)
