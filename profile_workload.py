"""Repeated search workload long enough to profile with py-spy and Scalene."""

import sys
from pathlib import Path

from findex.inverted import build_index
from findex.model import Index, load_model
from findex.numpy_rank import ranked_search_numpy
from findex.ranking import BM25, ranked_search


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "search"
    if mode == "index":
        for _ in range(10):
            build_index(Path("data"))
        return
    model: Index = load_model("index.bin")
    try:
        for _ in range(1500):
            if mode == "numpy":
                ranked_search_numpy(model, "python OR база OR тестування", BM25(), 10)
            else:
                ranked_search(model, "python OR база OR тестування", BM25(), 10)
    finally:
        model.close()


if __name__ == "__main__":
    main()
