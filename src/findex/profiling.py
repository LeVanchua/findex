"""Reusable timing and allocation measurement helpers."""

from dataclasses import dataclass
from time import perf_counter
import tracemalloc
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass
class Profile:
    value: object
    seconds: float
    peak_bytes: int


def profile(func: Callable[[], T]) -> Profile:
    tracemalloc.start()
    try:
        started = perf_counter()
        value = func()
        elapsed = perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return Profile(value, elapsed, peak)
