"""Reproducible Lab 05 benchmark. Run: uv run python bench_lab05.py data/.

Runs each configuration in a fresh child interpreter to reset RSS maxima.
Linux CPU time includes parent + completed child processes (resource module).
"""

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
from time import perf_counter


def single(corpus: str, workers: int, executor: str) -> dict[str, float | int | str]:
    from findex.parallel import build_parallel

    before_self = resource.getrusage(resource.RUSAGE_SELF)
    before_child = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = perf_counter()
    index, merge_seconds = build_parallel(corpus, workers, executor)
    wall = perf_counter() - started
    after_self = resource.getrusage(resource.RUSAGE_SELF)
    after_child = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = sum((getattr(after_self, key) - getattr(before_self, key)) +
               (getattr(after_child, key) - getattr(before_child, key))
               for key in ("ru_utime", "ru_stime"))
    return {"executor": executor, "workers": workers, "wall_s": wall, "cpu_s": cpu,
            "parent_peak_rss_mb": after_self.ru_maxrss / 1024, "merge_s": merge_seconds,
            "docs": index.document_count, "terms": index.vocabulary_size}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", nargs="?", default="data")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", default="benchmark_lab05.json")
    parser.add_argument("--one", nargs=2, metavar=("EXECUTOR", "WORKERS"))
    parser.add_argument("--python", default=sys.executable, help="Interpreter used in child measurements (3.13t supported)")
    parser.add_argument("--only-executor", choices=["serial", "threads", "processes"])
    args = parser.parse_args()
    if args.one:
        print(json.dumps(single(args.corpus, int(args.one[1]), args.one[0])))
        return
    if args.repetitions < 3:
        raise SystemExit("At least 3 repetitions required")
    kinds = (args.only_executor,) if args.only_executor else ("serial", "threads", "processes")
    configurations = [(kind, workers) for kind in kinds
                      for workers in sorted({1, 2, 4, 8, os.cpu_count() or 1})]
    rows = []
    for kind, workers in configurations:
        samples = []
        for repeat in range(args.repetitions):
            command = [args.python, str(Path(__file__).resolve()), args.corpus,
                       "--one", kind, str(workers)]
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parent / "src"))
            output = subprocess.check_output(command, text=True, env=env)
            samples.append(json.loads(output))
            print(f"{kind:9} {workers:3} repetition {repeat + 1}/{args.repetitions}: "
                  f"{samples[-1]['wall_s']:.3f}s", flush=True)
        rows.append({"executor": kind, "workers": workers,
                     **{metric: statistics.median(sample[metric] for sample in samples)
                        for metric in ("wall_s", "cpu_s", "parent_peak_rss_mb", "merge_s")},
                     "docs": samples[0]["docs"], "terms": samples[0]["terms"]})
    baseline = next(row["wall_s"] for row in rows if row["executor"] == (args.only_executor or "serial") and row["workers"] == 1)
    for row in rows:
        row["speedup"] = baseline / row["wall_s"]
    model = "unknown"
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    report = {"machine": platform.platform(), "cpu_model": model, "logical_cpus": os.cpu_count(),
              "python": args.python, "gil_enabled": subprocess.check_output(
                  [args.python, "-c", 'import sys;print(getattr(sys,"_is_gil_enabled",lambda: True)())'],
                  text=True).strip(),
              "repetitions": args.repetitions, "rows": rows}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for kind in kinds:
            subset = [row for row in rows if row["executor"] == kind]
            plt.plot([row["workers"] for row in subset], [row["speedup"] for row in subset],
                     marker="o", label=kind)
        xs = sorted({row["workers"] for row in rows})
        plt.plot(xs, xs, "k--", label="ideal linear")
        plt.xlabel("Workers")
        plt.ylabel("Speedup vs serial/1")
        plt.title("findex: parallel indexing speedup")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(args.output).with_suffix(".png"), dpi=160)
        plt.close()
    except ImportError:
        print("Matplotlib unavailable: install matplotlib and rerun to create graph", file=sys.stderr)


if __name__ == "__main__":
    main()
