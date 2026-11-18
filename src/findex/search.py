"""Usage: python -m findex.search index.bin 'python OR сервер' --engine merge."""

import argparse
from pathlib import Path

from .boolean import search
from .profiling import profile
from .serialization import load_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Search a saved findex inverted index")
    parser.add_argument("index", type=Path)
    parser.add_argument("query")
    parser.add_argument("--format", choices=["pickle", "json"], default="pickle")
    parser.add_argument("--engine", choices=["merge", "set"], default="merge")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    def action():
        index = load_index(args.index, args.format)
        return index, search(index, args.query, args.engine)
    try:
        measured = profile(action)
    except ValueError as error:
        parser.error(str(error))
    index, matching = measured.value
    print(f"Found {len(matching)} documents (showing first {max(0, args.limit)}):")
    for doc_id in matching[:max(0, args.limit)]:
        doc = index.documents[doc_id]
        print(f"  [{doc_id}] {doc.title} — {doc.path} ({doc.length} tokens)")
    print(f"Time (load + search): {measured.seconds:.4f} s; peak Python memory: {measured.peak_bytes / 1024**2:.3f} MiB")


if __name__ == "__main__":
    main()
