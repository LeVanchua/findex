"""Transparent profiling decorators with functools.wraps."""

import functools
import logging
from time import perf_counter
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)
logger = logging.getLogger("findex.timing")


def timed(function: F) -> F:
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        started = perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            logger.info("@timed %s %.6f s", function.__qualname__, perf_counter() - started)
    return wrapper  # type: ignore[return-value]
