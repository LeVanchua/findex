"""Command-line interface for the first findex laboratory."""

import argparse
from pathlib import Path

from .sample_data import generate_corpus
from .stats import measure


def main() -> None:
    parser = argparse.ArgumentParser(description="findex corpus profiler")
    sub = parser.add_subparsers(dest="command", required=True)
    generator = sub.add_parser("generate", help="create synthetic corpus under data/")
    generator.add_argument("--data", type=Path, default=Path("data"))
    generator.add_argument("--count", type=int, default=1000)
    generator.add_argument("--repetitions", type=int, default=20)
    report = sub.add_parser("benchmark", help="compare streaming and greedy")
    report.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()

    if args.command == "generate":
        generate_corpus(args.data, args.count, args.repetitions)
        print(f"Generated {args.count} documents in {args.data}")
        return

    streaming = measure("stream", args.data)
    greedy = measure("greedy", args.data)
    assert (streaming.stats.documents, streaming.stats.tokens, streaming.stats.term_counts) == (
        greedy.stats.documents, greedy.stats.tokens, greedy.stats.term_counts
    ), "Statistics from both modes must agree"
    print("Mode       Documents     Tokens    Vocabulary   Time (s)   Peak (MiB)")
    for result in (streaming, greedy):
        print(f"{result.mode:<10} {result.stats.documents:>9} {result.stats.tokens:>10} "
              f"{result.stats.vocabulary:>13} {result.elapsed_seconds:>10.4f} "
              f"{result.peak_bytes / 1024**2:>12.3f}")
    print("\nTop 50 terms (term: frequency):")
    for term, amount in streaming.stats.top_terms(50):
        print(f"{term}: {amount}")


if __name__ == "__main__":
    main()
