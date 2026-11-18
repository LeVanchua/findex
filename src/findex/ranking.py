"""Swappable TF-IDF/BM25 scorers, top-k ranking and highlighted snippets."""

from dataclasses import dataclass, field
import heapq
import math
from pathlib import Path
import re
from typing import Protocol
import logging

from .model import Index
from .query import parse_query
from .timing import timed

log = logging.getLogger("findex.search")


class Scorer(Protocol):
    def score(self, tf: int, df: int, doc_length: int, total_docs: int, avg_length: float) -> float: ...


@dataclass(frozen=True, slots=True)
class TfIdf:
    def score(self, tf: int, df: int, doc_length: int, total_docs: int, avg_length: float) -> float:
        if not tf or not df or not total_docs:
            return 0.0
        return (1 + math.log(tf)) * (math.log((total_docs + 1) / (df + 1)) + 1)


@dataclass(frozen=True, slots=True)
class BM25:
    k1: float = 1.5
    b: float = 0.75

    def score(self, tf: int, df: int, doc_length: int, total_docs: int, avg_length: float) -> float:
        if not tf or not df or not total_docs:
            return 0.0
        idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
        denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / max(avg_length, 1))
        return idf * tf * (self.k1 + 1) / denominator


@dataclass(order=True, frozen=True)
class SearchResult:
    score: float
    tie_breaker: int
    doc_id: int = field(compare=False)
    title: str = field(compare=False)
    path: str = field(compare=False)
    snippet: str = field(compare=False, default="")


def make_snippet(text: str, terms: tuple[str, ...], radius: int = 80) -> str:
    """Choose the earliest matching term; surround with ±radius characters.

    Highlight terms using **term**. The Markdown marker is output text only.
    """
    if not text:
        return ""
    pattern = re.compile("|".join(re.escape(t) for t in sorted(set(terms), key=len, reverse=True)), re.I) if terms else None
    match = pattern.search(text) if pattern else None
    if not match:
        return text[:radius * 2].replace("\n", " ")
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    fragment = text[start:end].replace("\n", " ")
    if pattern is not None:
        fragment = pattern.sub(lambda found: f"**{found.group()}**", fragment)
    return ("…" if start else "") + fragment + ("…" if end < len(text) else "")


@timed
def ranked_search(index: Index, query: str, scorer: Scorer | None = None,
                  top_k: int = 5, corpus: str | Path | None = None) -> list[SearchResult]:
    """Filter by Boolean AST, then sum positive-term ranking contributions."""
    scorer = scorer if scorer is not None else BM25()
    old = index.cached_ids.cache_info()
    ids = index.cached_ids(query)
    new = index.cached_ids.cache_info()
    log.info("query cache %s: %r", "HIT" if new.hits > old.hits else "MISS", query)
    node = parse_query(query)
    terms = tuple(dict.fromkeys(node.positive_terms()))
    results: list[SearchResult] = []
    for doc_id in ids:
        meta = index.documents[doc_id]
        score = 0.0
        for term in terms:
            # Sorted posting list ensures a stable, directly inspectable format.
            frequency = next((p.frequency for p in index.postings_for(term) if p.doc_id == doc_id), 0)
            if frequency:
                score += scorer.score(frequency, index.df(term), meta.length, index.num_docs, index.avg_doc_length)
        snippet = ""
        if corpus is not None:
            base = Path(corpus).resolve()
            candidate = (base / meta.path).resolve()
            if candidate.is_relative_to(base) and candidate.is_file():
                snippet = make_snippet(candidate.read_text(encoding="utf-8"), terms)
        results.append(SearchResult(score, -doc_id, doc_id, meta.title, meta.path, snippet))
    return heapq.nlargest(max(0, top_k), results)
