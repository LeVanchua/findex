"""Observed NumPy speedups and 10x memmap experiments (not fabricated data)."""

import json
from pathlib import Path
import resource
import statistics
import subprocess
import sys
from time import perf_counter

from findex.inverted import build_index
from findex.memmap_index import open_memmap, save_memmap
from findex.model import Index
from findex.numpy_rank import ranked_search_numpy
from findex.ranking import BM25, ranked_search
from findex.sample_data import generate_corpus
from findex.serialization import load_index, save_index


def median_ms(action: object, runs: int = 21) -> float:
    if not callable(action):
        raise ValueError("Action must be callable")
    for _ in range(3):
        action()
    times = []
    for _ in range(runs):
        start = perf_counter()
        action()
        times.append((perf_counter() - start) * 1000)
    return round(statistics.median(times), 4)


def benchmark_search() -> list[dict[str, object]]:
    model = Index(load_index("index.bin", "pickle"))
    model.numpy_index  # allocate upfront, as done once during web service lifespan
    samples = [
        ("частий термін", "python"),
        ("рідкісний термін", "999"),
        ("три терми", "python OR база OR тестування"),
    ]
    rows = []
    for label, query in samples:
        baseline = median_ms(lambda: ranked_search(model, query, BM25(), 10), 31)
        vector = median_ms(lambda: ranked_search_numpy(model, query, BM25(), 10), 31)
        rows.append({"case": label, "query": query, "python_ms": baseline,
                     "numpy_ms": vector, "speedup": round(baseline / vector, 3) if vector else 0})
    model.close()
    return rows


def load_measure(mode: str, directory: Path) -> dict[str, object]:
    start = perf_counter()
    if mode == "pickle":
        model = load_index(directory / "index.bin", "pickle")
        count = model.document_count
    else:
        model_map = open_memmap(directory / "mapped")
        count = len(model_map.lengths) - 1
        # Access a few postings so mapping is verified, without paging all terms in.
        model_map.postings_for("python")
    wall = round((perf_counter() - start) * 1000, 3)
    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"mode": mode, "documents": count, "load_ms": wall,
            "peak_process_rss_kib": peak_rss_kib}


def benchmark_memmap(directory: Path) -> list[dict[str, object]]:
    source = directory / "corpus"
    if not (source / "doc_10000.txt").exists():
        generate_corpus(source, 10_000, repetitions=2)
    index = build_index(source)
    save_index(index, directory / "index.bin", "pickle")
    save_memmap(index, directory / "mapped")
    del index
    rows = []
    for mode in ("pickle", "memmap"):
        process = subprocess.run([sys.executable, __file__, "--load", mode, str(directory)],
                                 capture_output=True, text=True, check=True)
        entry = json.loads(process.stdout)
        if mode == "pickle":
            size = (directory / "index.bin").stat().st_size
        else:
            size = sum(p.stat().st_size for p in (directory / "mapped").iterdir() if p.is_file())
        entry["storage_bytes"] = size
        rows.append(entry)
    return rows


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--load":
        print(json.dumps(load_measure(sys.argv[2], Path(sys.argv[3]))))
    else:
        output = {"scorer": benchmark_search(), "memmap_10x": benchmark_memmap(Path("docs/memmap_experiment"))}
        Path("benchmark_lab08.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(output, ensure_ascii=False, indent=2))
