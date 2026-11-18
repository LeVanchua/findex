"""One-pass inverted index with sorted postings and optional positions."""

from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .corpus import iter_documents
from .tokenizer import tokenize
from .timing import timed


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    frequency: int


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: str
    title: str
    length: int


@dataclass
class InvertedIndex:
    postings: dict[str, list[Posting]]
    documents: dict[int, DocMeta]
    positions: dict[str, dict[int, list[int]]] | None = None

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def vocabulary_size(self) -> int:
        return len(self.postings)

    def top_terms(self, limit: int = 50) -> list[tuple[str, int]]:
        return sorted(((term, sum(p.frequency for p in ps)) for term, ps in self.postings.items()),
                      key=lambda item: (-item[1], item[0]))[:limit]


@timed
def build_index(root: str | Path, with_positions: bool = False) -> InvertedIndex:
    """Traverse document generator exactly once; doc IDs increase from 1."""
    accumulator: defaultdict[str, list[Posting]] = defaultdict(list)
    metadata: dict[int, DocMeta] = {}
    positions: defaultdict[str, dict[int, list[int]]] | None = defaultdict(dict) if with_positions else None
    for doc_id, document in enumerate(iter_documents(root), 1):
        counts: Counter[str] = Counter()
        length = 0
        for offset, token in enumerate(tokenize(document.text)):
            counts[token] += 1
            length += 1
            if positions is not None:
                positions[token].setdefault(doc_id, []).append(offset)
        metadata[doc_id] = DocMeta(doc_id, str(document.path.relative_to(root)), document.path.stem, length)
        for term, frequency in counts.items():
            accumulator[term].append(Posting(doc_id, frequency))
    return InvertedIndex(dict(accumulator), metadata, dict(positions) if positions is not None else None)


@dataclass
class PlainPosting:
    doc_id: int
    frequency: int


def repr_memory(root: str | Path, variant: str) -> tuple[object, int]:
    """Build complete comparable posting representation for profiling.

    Returns both structure and number of posting entries so it stays alive
    during tracemalloc's peak observation.
    """
    if variant not in {"plain", "slots", "array"}:
        raise ValueError(variant)
    regular: defaultdict[str, list[PlainPosting | Posting]] = defaultdict(list)
    compact: defaultdict[str, tuple[array[int], array[int]]] = defaultdict(lambda: (array("I"), array("I")))
    total = 0
    for doc_id, document in enumerate(iter_documents(root), 1):
        counts = Counter(tokenize(document.text))
        for term, frequency in counts.items():
            total += 1
            if variant == "array":
                docs, freqs = compact[term]
                docs.append(doc_id)
                freqs.append(frequency)
            elif variant == "plain":
                regular[term].append(PlainPosting(doc_id, frequency))
            else:
                regular[term].append(Posting(doc_id, frequency))
    return (compact if variant == "array" else regular), total
