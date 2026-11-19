"""Safe localhost-only benchmark: 240 synthetic pages, no external crawling."""

import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import resource
import statistics
from threading import Thread
from time import perf_counter, sleep

from findex.crawler import CrawlStats, crawl


class Mirror(BaseHTTPRequestHandler):
    errors_once: set[str] = set()

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        if self.path == "/robots.txt":
            body = b"User-agent: *\nAllow: /\n"
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)
            return
        sleep(0.025)  # Simulates I/O waiting, not compute-intensive work.
        if self.path == "/p/17" and self.path not in self.errors_once:
            self.errors_once.add(self.path)
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.end_headers()
            return
        links = "".join(f'<a href="/p/{n}">document {n}</a>' for n in range(1, 240)) if self.path == "/" else ""
        body = f"<html><body><h1>Python asynchronous sample {self.path}</h1>{links}</body></html>".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


async def measure(url: str, concurrency: int, log_file: Path) -> dict[str, float | int]:
    import logging

    crawl_logger = logging.getLogger("findex.crawl")
    handler = logging.FileHandler(log_file, encoding="utf-8", mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    crawl_logger.addHandler(handler)
    crawl_logger.setLevel(logging.INFO)
    stats = CrawlStats()
    started = perf_counter()
    try:
        pages = 0
        async for result in crawl(url, max_pages=200, concurrency=concurrency,
                                  per_host=20, host_delay=0.003, stats=stats):
            if result.status == 200:
                pages += 1
        wall = perf_counter() - started
        return {"concurrency": concurrency, "wall_s": wall, "pages": pages,
                "pages_per_s": pages / wall, "errors": stats.errors,
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    finally:
        crawl_logger.removeHandler(handler)
        handler.close()


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Mirror)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        start_url = f"http://127.0.0.1:{server.server_port}/"
        rows = []
        for concurrency in (1, 5, 20):
            trials = []
            for repeat in range(3):
                Mirror.errors_once.clear()  # Inject one fresh 429 in each independent run.
                row = asyncio.run(measure(start_url, concurrency, Path("crawl.log") if concurrency == 20 and repeat == 2
                                          else Path("benchmark_crawl.log")))
                trials.append(row)
                print(f"concurrency={concurrency} repeat={repeat + 1} "
                      f"pages={row['pages']} time={row['wall_s']:.3f}s")
            rows.append({"concurrency": concurrency, "pages": trials[0]["pages"],
                         "wall_s": statistics.median(t["wall_s"] for t in trials),
                         "pages_per_s": statistics.median(t["pages_per_s"] for t in trials),
                         "errors": statistics.median(t["errors"] for t in trials),
                         "peak_rss_mib": max(t["peak_rss_mib"] for t in trials)})
        Path("benchmark_lab06.json").write_text(json.dumps({
            "permission": "Local HTTP server created and owned by this benchmark; robots.txt allows /",
            "page_limit": 200, "site_pages": 240, "host_delay": 0.003,
            "per_host": 20, "trials_per_configuration": 3, "rows": rows,
        }, indent=2), encoding="utf-8")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


if __name__ == "__main__":
    main()
