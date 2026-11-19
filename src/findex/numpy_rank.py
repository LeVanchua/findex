"""Vectorized BM25/TF-IDF with compact int32 postings and deterministic top-k."""
# pyright: reportUnknownMemberType=false

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .model import Index
from .query import parse_query
from .ranking import BM25, Scorer, SearchResult, TfIdf, make_snippet
from .ranking import _jsonl_snippet as jsonl_snippet  # pyright: ignore[reportPrivateUsage]

IntArray = NDArray[np.int32]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class NumPyIndex:
    """Term postings and document lengths in contiguous NumPy arrays."""

    postings: dict[str, tuple[IntArray, IntArray]]
    lengths: IntArray

    @classmethod
    def from_index(cls, index: Index) -> "NumPyIndex":
        if index.documents and max(index.documents) > np.iinfo(np.int32).max:
            raise OverflowError("Document ID exceeds NumPy int32 range")
        max_id = max(index.documents, default=0)
        lengths = np.zeros(max_id + 1, dtype=np.int32)
        for doc_id, meta in index.documents.items():
            lengths[doc_id] = meta.length
        postings = {
            term: (
                np.fromiter((posting.doc_id for posting in index[term]), dtype=np.int32),
                np.fromiter((posting.frequency for posting in index[term]), dtype=np.int32),
            )
            for term in index
        }
        return cls(postings, lengths)


def vector_scores(index: Index, query: str, scorer: Scorer | None = None) -> tuple[IntArray, FloatArray]:
    """Vectorized score for matching docs; Boolean filtering remains unchanged."""
    scorer = scorer if scorer is not None else BM25()
    if not isinstance(scorer, (BM25, TfIdf)):
        raise ValueError("NumPy scorer supports BM25 or TfIdf")
    ids = np.asarray(index.cached_ids(query), dtype=np.int32)
    if len(ids) == 0:
        return ids, np.zeros(0, dtype=np.float64)
    vectors = index.numpy_index
    scores = np.zeros(len(ids), dtype=np.float64)
    terms = tuple(dict.fromkeys(parse_query(query).positive_terms()))
    for term in terms:
        term_data = vectors.postings.get(term)
        if term_data is None:
            continue
        docs, freqs = term_data
        if len(docs) == 0:
            continue
        # searchsorted is O(log n) per posting and avoids Python object loops.
        positions = np.searchsorted(ids, docs)
        inside = positions < len(ids)
        selected = np.zeros(len(docs), dtype=np.bool_)
        selected[inside] = ids[positions[inside]] == docs[inside]
        if not np.any(selected):
            continue
        indices = positions[selected]
        tf = freqs[selected].astype(np.float64)
        df = len(docs)
        if isinstance(scorer, BM25):
            idf = math.log(1 + (index.num_docs - df + .5) / (df + .5))
            lengths = vectors.lengths[docs[selected]].astype(np.float64)
            denominator = tf + scorer.k1 * (1 - scorer.b + scorer.b * lengths / max(index.avg_doc_length, 1))
            contribution = idf * tf * (scorer.k1 + 1) / denominator
        else:
            contribution = (1 + np.log(tf)) * (math.log((index.num_docs + 1) / (df + 1)) + 1)
        np.add.at(scores, indices, contribution)
    return ids, scores


def top_k_indices(ids: IntArray, scores: FloatArray, k: int) -> IntArray:
    """Use argpartition while retaining all boundary ties (ascending doc_id)."""
    if k < 1 or len(ids) == 0:
        return np.zeros(0, dtype=np.int32)
    if k >= len(ids):
        pool = np.arange(len(ids))
    else:
        partition = np.argpartition(scores, -k)[-k:]
        threshold = float(np.min(scores[partition]))
        pool = np.flatnonzero(scores >= threshold)
    ordered = np.lexsort((ids[pool], -scores[pool]))
    return np.asarray(pool[ordered[:k]], dtype=np.int32)


def ranked_search_numpy(
    index: Index, query: str, scorer: Scorer | None = None,
    top_k: int = 5, corpus: str | Path | None = None,
) -> list[SearchResult]:
    """NumPy ranking with legacy-compatible SearchResult and snippet formatting."""
    ids, scores = vector_scores(index, query, scorer)
    selected = top_k_indices(ids, scores, top_k)
    terms = tuple(dict.fromkeys(parse_query(query).positive_terms()))
    results: list[SearchResult] = []
    for position in selected:
        doc_id = int(ids[position])
        meta = index.documents[doc_id]
        snippet = ""
        if corpus is not None:
            base = Path(corpus).resolve()
            candidate = (base / meta.path).resolve()
            if candidate.is_relative_to(base) and candidate.is_file():
                snippet = make_snippet(candidate.read_text(encoding="utf-8"), terms)
            elif candidate.is_relative_to(base):
                snippet = jsonl_snippet(base, meta.path, terms)
        results.append(SearchResult(float(scores[position]), -doc_id, doc_id, meta.title, meta.path, snippet))
    return results
