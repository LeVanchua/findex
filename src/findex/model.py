"""Pythonic inverted index façade and a safe-lifetime context manager."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Generator

from .inverted import DocMeta, InvertedIndex, Posting
from .serialization import load_index
from .timing import timed


class Index(Mapping[str, tuple[Posting, ...]]):
    """Read-only Mapping: term -> tuple of sorted postings.

Use open_index(path) for deterministic cleanup of transient query caches.
"""

    def __hash__(self) -> int:
        return object.__hash__(self)  # identity scopes the LRU cache to this index

    def __init__(self, raw: InvertedIndex):
        self._data = raw
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Index context is already closed")

    def __len__(self) -> int:
        self._check_open()
        return len(self._data.postings)

    def __iter__(self) -> Iterator[str]:
        self._check_open()
        return iter(self._data.postings)

    def __getitem__(self, term: str) -> tuple[Posting, ...]:
        self._check_open()
        return tuple(self._data.postings[term])

    def __contains__(self, term: object) -> bool:
        self._check_open()
        return term in self._data.postings

    def __repr__(self) -> str:
        if self._closed:
            return "Index(closed=True)"
        return f"Index(num_docs={self.num_docs}, terms={len(self)})"

    @property
    def num_docs(self) -> int:
        self._check_open()
        return len(self._data.documents)

    @cached_property
    def avg_doc_length(self) -> float:
        self._check_open()
        return sum(m.length for m in self._data.documents.values()) / max(1, self.num_docs)

    def df(self, term: str) -> int:
        self._check_open()
        return len(self._data.postings.get(term, ()))

    @property
    def documents(self) -> dict[int, DocMeta]:
        self._check_open()
        return self._data.documents

    @property
    def positions(self) -> dict[str, dict[int, list[int]]] | None:
        self._check_open()
        return self._data.positions

    @lru_cache(maxsize=512)
    def cached_ids(self, query: str) -> tuple[int, ...]:
        """LRU memoization by query text for the current index instance."""
        from .query import parse_query
        self._check_open()
        return tuple(sorted(parse_query(query).evaluate(self)))

    def close(self) -> None:
        self.cached_ids.cache_clear()
        self.__dict__.pop("avg_doc_length", None)
        self._closed = True

    def postings_for(self, term: str) -> tuple[Posting, ...]:
        return self[term] if term in self else ()


@timed
def load_model(path: str | Path, fmt: str = "pickle") -> Index:
    """Load an index that was saved by the trusted local CLI."""
    return Index(load_index(path, fmt))


@contextmanager
def open_index(path: str | Path, fmt: str = "pickle") -> Generator[Index, None, None]:
    index = load_model(path, fmt)
    try:
        yield index
    finally:
        index.close()
