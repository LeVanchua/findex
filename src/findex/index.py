"""Usage: python -m findex.index data/ --out index.bin [--format pickle|json]."""

import argparse
from pathlib import Path

from .inverted import build_index
from .profiling import profile
from .serialization import save_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the findex inverted index")
    parser.add_argument("data", type=Path, help="directory with UTF-8 .txt files")
    parser.add_argument("--out", type=Path, default=Path("index.bin"))
    parser.add_argument("--format", choices=["pickle", "json"], default="pickle")
    parser.add_argument("--positions", action="store_true", help="store token offsets (larger index)")
    args = parser.parse_args()
    def action():
        result = build_index(args.data, with_positions=args.positions)
        save_index(result, args.out, args.format)
        return result
    measured = profile(action)
    index = measured.value
    print(f"Documents: {index.document_count}; vocabulary: {index.vocabulary_size}")
    print(f"Saved: {args.out} ({args.format}, {args.out.stat().st_size} bytes)")
    print(f"Time: {measured.seconds:.4f} s; peak Python memory: {measured.peak_bytes / 1024**2:.3f} MiB")


if __name__ == "__main__":
    main()
