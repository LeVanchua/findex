"""Reproducible 20-concurrent-request local API latency benchmark.

When oha is installed it should be run separately for the formal lab table.
This script uses httpx and explicitly labels its results (not oha).
"""

import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from time import perf_counter, sleep

import httpx


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (position - lower)


async def measure(url: str, requests: int = 240, concurrency: int = 20) -> dict[str, float | int]:
    timings: list[float] = []
    failures = 0
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=20, trust_env=False) as client:
        async def one() -> None:
            nonlocal failures
            async with sem:
                started = perf_counter()
                try:
                    response = await client.get(url)
                    if response.status_code != 200:
                        failures += 1
                except httpx.HTTPError:
                    failures += 1
                finally:
                    timings.append((perf_counter() - started) * 1000)

        started = perf_counter()
        await asyncio.gather(*(one() for _ in range(requests)))
        wall = perf_counter() - started
    return {
        "requests": requests, "concurrency": concurrency, "errors": failures,
        "qps": round(requests / wall, 2),
        "p50_ms": round(percentile(timings, .50), 2),
        "p95_ms": round(percentile(timings, .95), 2),
        "p99_ms": round(percentile(timings, .99), 2),
    }


def wait_ready(port: int) -> None:
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=.2):
                return
        except OSError:
            sleep(.1)
    raise TimeoutError("Test Uvicorn server did not start")


def run_case(label: str, workers: int, route: str) -> dict[str, str | float | int]:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = dict(os.environ, INDEX_PATH=str(Path("index.bin").resolve()), CORPUS_PATH=str(Path("data").resolve()))
    command = [sys.executable, "-m", "uvicorn", "findex.web.app:app", "--host", "127.0.0.1", "--port", str(port), "--workers", str(workers), "--no-access-log"]
    server = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready(port)
        url = f"http://127.0.0.1:{port}{route}?q=python&k=10"
        result = asyncio.run(measure(url))
        return {"case": label, "workers": workers, **result}
    finally:
        server.terminate()
        try:
            server.wait(timeout=6)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


if __name__ == "__main__":
    output = [
        run_case("1 worker · sync def", 1, "/search"),
        run_case("1 worker · async def", 1, "/bench/async-search"),
        run_case("4 workers · sync def", 4, "/search"),
    ]
    Path("benchmark_lab07.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
