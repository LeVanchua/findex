"""Regression tests for vectorized ranking and stable boundary ties."""

from pathlib import Path

import numpy as np
import pytest

from findex.inverted import build_index
from findex.model import Index
from findex.numpy_rank import ranked_search_numpy, top_k_indices
from findex.ranking import BM25, TfIdf, ranked_search
from findex.sample_data import generate_corpus


@pytest.fixture(scope="module")
def index(tmp_path_factory: pytest.TempPathFactory) -> Index:
    corpus = tmp_path_factory.mktemp("numpy_corpus")
    generate_corpus(corpus, count=120, repetitions=2)
    return Index(build_index(corpus))


@pytest.mark.parametrize("query", [
    "python", "індекс", "python OR база OR тестування",
    "python AND генератори", "тестування OR пошук", "алгоритм",
])
@pytest.mark.parametrize("scorer", [BM25(), TfIdf()])
def test_numpy_order_matches_original(index: Index, query: str, scorer: BM25 | TfIdf) -> None:
    expected = ranked_search(index, query, scorer, 50)
    actual = ranked_search_numpy(index, query, scorer, 50)
    assert [item.doc_id for item in actual] == [item.doc_id for item in expected]
    assert [item.score for item in actual] == pytest.approx([item.score for item in expected])


def test_numpy_postings_are_int32(index: Index) -> None:
    ids, frequencies = index.numpy_index.postings["python"]
    assert ids.dtype == np.int32
    assert frequencies.dtype == np.int32
    assert index.numpy_index.lengths.dtype == np.int32


def test_argpartition_breaks_boundary_ties_by_doc_id() -> None:
    ids = np.array([11, 3, 9, 1], dtype=np.int32)
    scores = np.array([.5, .5, .5, .5], dtype=np.float64)
    positions = top_k_indices(ids, scores, 2)
    assert list(ids[positions]) == [1, 3]


def test_snippets_identical(tmp_path: Path) -> None:
    generate_corpus(tmp_path, count=10, repetitions=1)
    index = Index(build_index(tmp_path))
    old = ranked_search(index, "python", BM25(), 5, tmp_path)
    new = ranked_search_numpy(index, "python", BM25(), 5, tmp_path)
    assert [r.snippet for r in new] == [r.snippet for r in old]
