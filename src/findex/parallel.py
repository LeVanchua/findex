"""Deterministic parallel indexing: workers read paths, parent merges results."""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from multiprocessing import get_context
from time import perf_counter
from collections.abc import Iterator
from typing import Literal

from .inverted import DocMeta, InvertedIndex, Posting
from .tokenizer import tokenize

ExecutorKind = Literal["serial", "threads", "processes"]


def corpus_paths(root: str | Path) -> list[str]:
    """List relative file paths in the original generator's deterministic order."""
    base = Path(root)
    return [str(path.relative_to(base)) for path in iter_document_paths(base)]


def iter_document_paths(root: Path) -> Iterator[Path]:
    """Enumerate paths without reading document bodies."""
    import os

    if not root.is_dir():
        raise NotADirectoryError(f"Corpus directory does not exist: {root}")
    for directory, dirs, files in os.walk(root):
        dirs.sort()
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix.lower() == ".txt":
                yield path


def build_partial(root: str | Path, paths: list[str], start_id: int, with_positions: bool = False) -> InvertedIndex:
    """Spawn-safe top-level worker; reads files itself and owns a disjoint ID range."""
    base = Path(root)
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    docs: dict[int, DocMeta] = {}
    positions: defaultdict[str, dict[int, list[int]]] | None = defaultdict(dict) if with_positions else None
    for offset, relative in enumerate(paths):
        doc_id = start_id + offset
        path = base / relative
        counts: Counter[str] = Counter()
        length = 0
        for position, token in enumerate(tokenize(path.read_text(encoding="utf-8"))):
            counts[token] += 1
            length += 1
            if positions is not None:
                positions[token].setdefault(doc_id, []).append(position)
        docs[doc_id] = DocMeta(doc_id, relative, path.stem, length)
        for term, frequency in counts.items():
            postings[term].append(Posting(doc_id, frequency))
    return InvertedIndex(dict(postings), docs, dict(positions) if positions is not None else None)


def merge(parts: list[InvertedIndex]) -> InvertedIndex:
    """Merge disjoint shards into an index with sorted posting lists."""
    combined: defaultdict[str, list[Posting]] = defaultdict(list)
    docs: dict[int, DocMeta] = {}
    positions: defaultdict[str, dict[int, list[int]]] | None = (
        defaultdict(dict) if any(p.positions is not None for p in parts) else None
    )
    for part in parts:
        if docs.keys() & part.documents.keys():
            raise ValueError("Duplicate document IDs in partial indexes")
        docs.update(part.documents)
        for term, values in part.postings.items():
            combined[term].extend(values)
        if positions is not None and part.positions is not None:
            for term, mapping in part.positions.items():
                positions[term].update(mapping)
    return InvertedIndex(
        {term: sorted(values, key=lambda posting: posting.doc_id) for term, values in combined.items()},
        dict(sorted(docs.items())),
        dict(positions) if positions is not None else None,
    )


def _chunks(paths: list[str], workers: int) -> list[tuple[list[str], int]]:
    size, remainder = divmod(len(paths), workers)
    cursor = 0
    chunks: list[tuple[list[str], int]] = []
    for worker in range(workers):
        count = size + (worker < remainder)
        if count:
            chunks.append((paths[cursor: cursor + count], cursor + 1))
        cursor += count
    return chunks


def build_parallel(
    root: str | Path,
    workers: int = 1,
    executor: ExecutorKind = "serial",
    with_positions: bool = False,
) -> tuple[InvertedIndex, float]:
    """Return index and merge seconds; .result() propagates worker exceptions."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if executor not in ("serial", "threads", "processes"):
        raise ValueError(f"Unknown executor: {executor}")
    base = Path(root).resolve()
    chunks = _chunks(corpus_paths(base), workers)
    if executor == "serial":
        parts = [build_partial(base, paths, start, with_positions) for paths, start in chunks]
    else:
        if executor == "threads":
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(build_partial, base, paths, start, with_positions) for paths, start in chunks]
                parts = [future.result() for future in futures]
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
                futures = [pool.submit(build_partial, base, paths, start, with_positions) for paths, start in chunks]
                parts = [future.result() for future in futures]
    started = perf_counter()
    result = merge(parts)
    return result, perf_counter() - started
