"""Logging helpers shared by the CLI and the library."""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager

_FMT = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(verbosity: int = 0, *, quiet: bool = False) -> None:
    """Configure root logging once, honouring ``KMER_DUST_LOGLEVEL`` if set."""
    env = os.environ.get("KMER_DUST_LOGLEVEL")
    if env:
        level = getattr(logging, env.upper(), logging.INFO)
    elif quiet:
        level = logging.WARNING
    else:
        level = {0: logging.INFO, 1: logging.DEBUG}.get(verbosity, logging.DEBUG)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    root.addHandler(handler)
    root.setLevel(level)
    # numba/urllib3 are chatty at DEBUG and never say anything we want.
    for noisy in ("numba", "urllib3", "matplotlib", "numexpr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@contextmanager
def timed(logger: logging.Logger, what: str):
    """Log ``what`` with its wall-clock duration."""
    start = time.perf_counter()
    logger.info("%s ...", what)
    try:
        yield
    finally:
        logger.info("%s done in %.1fs", what, time.perf_counter() - start)
