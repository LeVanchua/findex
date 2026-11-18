"""One-pass corpus statistics and comparative memory profiling."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import tracemalloc

from .corpus import iter_documents
from .tokenizer import tokenize


@dataclass
class CorpusStats:
    documents: int
    tokens: int
    term_counts: Counter[str]

    @property
    def vocabulary(self) -> int:
        return len(self.term_counts)

    def top_terms(self, n: int = 50) -> list[tuple[str, int]]:
        return sorted(self.term_counts.items(), key=lambda item: (-item[1], item[0]))[:n]


def streaming_stats(root: Path | str) -> CorpusStats:
    counts: Counter[str] = Counter()
    documents = tokens = 0
    for document in iter_documents(root):
        documents += 1
        for token in tokenize(document.text):
            counts[token] += 1
            tokens += 1
    return CorpusStats(documents, tokens, counts)


def greedy_stats(root: Path | str) -> CorpusStats:
    """Deliberately inefficient baseline: all docs and token lists coexist."""
    documents = list(iter_documents(root))
    token_lists = [list(tokenize(document.text)) for document in documents]
    all_tokens = [token for tokens in token_lists for token in tokens]
    return CorpusStats(len(documents), len(all_tokens), Counter(all_tokens))


@dataclass(frozen=True)
class Measurement:
    mode: str
    elapsed_seconds: float
    peak_bytes: int
    stats: CorpusStats


def measure(mode: str, root: Path | str) -> Measurement:
    if mode not in {"stream", "greedy"}:
        raise ValueError("mode must be 'stream' or 'greedy'")
    # Start/stop separately for comparable independent tracemalloc peaks.
    tracemalloc.start()
    try:
        start = perf_counter()
        stats = (streaming_stats if mode == "stream" else greedy_stats)(root)
        elapsed = perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return Measurement(mode, elapsed, peak, stats)
