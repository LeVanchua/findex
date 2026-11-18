"""Ten explicitly annotated queries for the deterministic synthetic corpus.

Each annotation names a relevant topic template, not engine-selected IDs.
The lab corpus repeats ten manually reviewed topic templates cyclically.
"""

from dataclasses import dataclass
from pathlib import Path

from .inverted import build_index
from .model import Index
from .ranking import BM25, TfIdf, ranked_search


@dataclass(frozen=True)
class Annotation:
    query: str
    relevant_topic: int  # 0-based topic in sample_data.TOPICS


ANNOTATIONS: tuple[Annotation, ...] = (
    Annotation("генератори", 0),
    Annotation("sql", 1),
    Annotation("тестування", 2),
    Annotation("протокол", 3),
    Annotation("навчання", 4),
    Annotation("алгоритм", 5),
    Annotation("вебзастосунок", 6),
    Annotation("git", 7),
    Annotation("unicode", 8),
    Annotation("корпус", 9),
)


def evaluate(index: Index) -> list[tuple[str, float, float]]:
    """P@5, with unreturned slots counted as non-relevant."""
    output = []
    for annotation in ANNOTATIONS:
        relevant = {doc_id for doc_id in index.documents if (doc_id - 1) % 10 == annotation.relevant_topic}
        scores = []
        for model in (TfIdf(), BM25()):
            top = ranked_search(index, annotation.query, model, top_k=5)
            scores.append(sum(result.doc_id in relevant for result in top) / 5)
        output.append((annotation.query, scores[0], scores[1]))
    return output


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate P@5 on sample topic annotations")
    parser.add_argument("data", type=Path, nargs="?", default=Path("data"))
    args = parser.parse_args()
    index = Index(build_index(args.data, with_positions=True))
    print("Query | TF-IDF P@5 | BM25 P@5")
    scores = evaluate(index)
    for query, tfidf, bm25 in scores:
        print(f"{query} | {tfidf:.2f} | {bm25:.2f}")
    print(f"AVERAGE | {sum(x[1] for x in scores)/len(scores):.2f} | {sum(x[2] for x in scores)/len(scores):.2f}")


if __name__ == "__main__":
    main()
