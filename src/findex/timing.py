"""Transparent profiling decorators with functools.wraps."""

import functools
import logging
from time import perf_counter
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")
logger = logging.getLogger("findex.timing")


def timed(function: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        started = perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            logger.info("@timed %s %.6f s", function.__qualname__, perf_counter() - started)
    return wrapper
