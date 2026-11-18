"""Reusable timing and allocation measurement helpers."""

from dataclasses import dataclass
from time import perf_counter
import tracemalloc
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass
class Profile(Generic[T]):
    value: T
    seconds: float
    peak_bytes: int


def profile(func: Callable[[], T]) -> Profile[T]:
    tracemalloc.start()
    try:
        started = perf_counter()
        value = func()
        elapsed = perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return Profile(value, elapsed, peak)
