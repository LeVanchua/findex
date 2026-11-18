"""Command line example with a phrase, ranking and a reusable index file."""

import argparse
import logging
from pathlib import Path

from .model import open_index
from .ranking import BM25, TfIdf, ranked_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Ranked Boolean/phrase search")
    parser.add_argument("index", type=Path)
    parser.add_argument("query")
    parser.add_argument("--format", choices=["pickle", "json"], default="pickle")
    parser.add_argument("--scorer", choices=["bm25", "tfidf"], default="bm25")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--corpus", type=Path, default=Path("data"))
    parser.add_argument("--repeat", action="store_true", help="run twice to demonstrate LRU cache hit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    scorer = BM25() if args.scorer == "bm25" else TfIdf()
    with open_index(args.index, args.format) as index:
        for run in range(2 if args.repeat else 1):
            print(f"Run {run + 1}; {index!r}; average length {index.avg_doc_length:.2f}")
            for item in ranked_search(index, args.query, scorer, args.top, args.corpus):
                print(f"  #{item.doc_id} score={item.score:.5f} {item.title} — {item.path}")
                if item.snippet:
                    print(f"     {item.snippet}")


if __name__ == "__main__":
    main()
