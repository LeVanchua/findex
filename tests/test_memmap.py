"""Disk-backed postings can round-trip on an enlarged corpus."""

from pathlib import Path

import numpy as np

from findex.inverted import build_index
from findex.memmap_index import open_memmap, save_memmap
from findex.sample_data import generate_corpus


def test_memmap_roundtrip(tmp_path: Path) -> None:
    source = tmp_path / "data"
    generate_corpus(source, 100, 1)
    original = build_index(source)
    target = tmp_path / "mapped"
    save_memmap(original, target)
    mapped = open_memmap(target)
    assert isinstance(mapped.doc_ids, np.memmap)
    assert mapped.doc_ids.dtype == np.int32
    assert mapped.lengths.dtype == np.int32
    for term in ("python", "індекс", "тестування", "missing"):
        doc_ids, frequencies = mapped.postings_for(term)
        expected = original.postings.get(term, [])
        assert list(doc_ids) == [item.doc_id for item in expected]
        assert list(frequencies) == [item.frequency for item in expected]
