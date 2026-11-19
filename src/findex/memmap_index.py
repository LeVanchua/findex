"""Disk-backed postings: contiguous int32 arrays opened through NumPy memmap."""
# pyright: reportUnknownMemberType=false

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .inverted import InvertedIndex

IntArray = NDArray[np.int32]


@dataclass(slots=True)
class MemmapPostings:
    """Metadata loads to RAM; postings stay memory-mapped on disk."""

    doc_ids: IntArray
    frequencies: IntArray
    lengths: IntArray
    offsets: dict[str, tuple[int, int]]

    def postings_for(self, term: str) -> tuple[IntArray, IntArray]:
        start, stop = self.offsets.get(term, (0, 0))
        return self.doc_ids[start:stop], self.frequencies[start:stop]


def save_memmap(index: InvertedIndex, directory: str | Path) -> None:
    """Write 3 .npy arrays and a compact offset table; no pickle loading."""
    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=True)
    max_id = max(index.documents, default=0)
    if max_id > np.iinfo(np.int32).max:
        raise OverflowError("Document IDs must fit in signed int32")
    lengths = np.zeros(max_id + 1, dtype=np.int32)
    for doc_id, meta in index.documents.items():
        lengths[doc_id] = meta.length
    doc_ids: list[int] = []
    frequencies: list[int] = []
    offsets: dict[str, tuple[int, int]] = {}
    for term in sorted(index.postings):
        start = len(doc_ids)
        for posting in index.postings[term]:
            doc_ids.append(posting.doc_id)
            frequencies.append(posting.frequency)
        offsets[term] = (start, len(doc_ids))
    np.save(dest / "doc_ids.npy", np.asarray(doc_ids, dtype=np.int32), allow_pickle=False)
    np.save(dest / "frequencies.npy", np.asarray(frequencies, dtype=np.int32), allow_pickle=False)
    np.save(dest / "lengths.npy", lengths, allow_pickle=False)
    (dest / "offsets.json").write_text(json.dumps(offsets, ensure_ascii=False), encoding="utf-8")


def open_memmap(directory: str | Path) -> MemmapPostings:
    """Memory-map posting arrays without bringing the entire index into RAM."""
    root = Path(directory)
    raw: dict[str, list[int]] = json.loads((root / "offsets.json").read_text(encoding="utf-8"))
    offsets = {term: (int(pair[0]), int(pair[1])) for term, pair in raw.items()}
    return MemmapPostings(
        np.load(root / "doc_ids.npy", mmap_mode="r", allow_pickle=False),
        np.load(root / "frequencies.npy", mmap_mode="r", allow_pickle=False),
        np.load(root / "lengths.npy", mmap_mode="r", allow_pickle=False),
        offsets,
    )
