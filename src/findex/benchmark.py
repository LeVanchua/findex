"""Reproducible memory, query and serialization comparisons."""

import argparse
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

from .boolean import search
from .inverted import build_index, repr_memory
from .profiling import profile
from .serialization import load_index, save_index


def benchmark(data: Path, repetitions: int = 100) -> None:
    print("POSTING REPRESENTATIONS (build, tracemalloc)")
    for variant in ("plain", "slots", "array"):
        p = profile(lambda v=variant: repr_memory(data, v))
        print(f"{variant:>7} | posting entries={p.value[1]} | {p.seconds:.4f} s | {p.peak_bytes/1024**2:.3f} MiB")
    index = build_index(data)
    top = sorted(index.postings, key=lambda t: (-sum(p.frequency for p in index.postings[t]), t))[:2]
    rare = sorted(index.postings, key=lambda t: (sum(p.frequency for p in index.postings[t]), t))[:2]
    print("\nSEARCH (median of repeated calls, index already in memory)")
    for name, terms in (("common", top), ("rare", rare)):
        query = " ".join(terms)
        for engine in ("merge", "set"):
            times = []
            for _ in range(repetitions):
                begin = perf_counter()
                result = search(index, query, engine)
                times.append((perf_counter() - begin) * 1000)
            print(f"{name:>6} | {engine:>5} | {query!r} | matches={len(result)} | median={median(times):.5f} ms")
    print("\nSERIALIZATION (file size, save and load profile)")
    with TemporaryDirectory() as folder:
        for fmt in ("pickle", "json"):
            file = Path(folder) / f"index.{fmt}"
            saved = profile(lambda f=file, format=fmt: save_index(index, f, format))
            loaded = profile(lambda f=file, format=fmt: load_index(f, format))
            assert loaded.value.postings == index.postings
            print(f"{fmt:>6} | {file.stat().st_size} bytes | save={saved.seconds:.4f} s | load={loaded.seconds:.4f} s | save_peak={saved.peak_bytes/1024**2:.3f} MiB | load_peak={loaded.peak_bytes/1024**2:.3f} MiB")
        plain = profile(lambda: build_index(data))
        positional = profile(lambda: build_index(data, with_positions=True))
        save_index(plain.value, Path(folder) / "plain.pickle")
        save_index(positional.value, Path(folder) / "positions.pickle")
        a = (Path(folder) / "plain.pickle").stat().st_size
        b = (Path(folder) / "positions.pickle").stat().st_size
        print(f"\nPOSITIONS off={a} bytes, on={b} bytes, increase={(b/a-1)*100:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare findex Lab 02 implementations")
    parser.add_argument("data", type=Path, nargs="?", default=Path("data"))
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()
    benchmark(args.data, args.repetitions)


if __name__ == "__main__":
    main()
