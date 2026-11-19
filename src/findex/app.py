"""Installable findex CLI: Typer commands with Rich output and clean JSON."""

import asyncio
import json
import logging
import os
from pathlib import Path
import sys
from time import monotonic
from typing import Annotated, Literal

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import typer

from .parallel import ExecutorKind, build_parallel
from .inverted import build_index
from .crawler import CrawlStats, crawl
from .model import open_index
from .ranking import BM25, TfIdf, ranked_search
from .numpy_rank import ranked_search_numpy
from .memmap_index import save_memmap
from .serialization import save_index
from .stats import streaming_stats


app = typer.Typer(help="findex — streaming index and ranked search", no_args_is_help=True, pretty_exceptions_enable=False)
error_console = Console(stderr=True)
out_console = Console()


@app.callback()
def setup(
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True, help="Increase logging (-v, -vv)")] = 0,
) -> None:
    """Configure diagnostics once; logs are sent to stderr."""
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if verbose >= 2 else logging.INFO if verbose == 1 else logging.WARNING,
        format="%(levelname)s %(message)s",
        force=True,
    )


def fail(error: Exception) -> None:
    error_console.print(f"Error: {error}", style="red")
    raise typer.Exit(code=1)


@app.command("index")
def index_command(
    data: Annotated[Path, typer.Argument(help="Folder with UTF-8 .txt documents")],
    out: Annotated[Path, typer.Option("--out", help="Output index file")] = Path("index.bin"),
    format: Annotated[Literal["pickle", "json"], typer.Option("--format")] = "pickle",
    positions: Annotated[bool, typer.Option("--positions", help="Store term positions")] = False,
    workers: Annotated[int, typer.Option("--workers", min=1)] = min(8, os.cpu_count() or 1),
    executor: Annotated[ExecutorKind, typer.Option("--executor")] = "processes",
) -> None:
    """Build and save an index, showing a Rich progress indicator."""
    try:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=error_console) as progress:
            task = progress.add_task("Indexing documents...", total=None)
            if any(data.rglob("*.jsonl")):
                # JSONL pages are streamed as virtual documents by iter_documents.
                result = build_index(data, with_positions=positions)
                merge_seconds = 0.0
            else:
                result, merge_seconds = build_parallel(data, workers=workers, executor=executor, with_positions=positions)
            save_index(result, out, format)
            progress.update(task, description="Index saved")
        out_console.print(f"Indexed {result.document_count} documents; {result.vocabulary_size} terms → {out}")
        logging.getLogger("findex").info("merge_seconds=%.6f executor=%s workers=%d", merge_seconds, executor, workers)
    except (OSError, UnicodeError, ValueError) as error:
        fail(error)


@app.command("crawl")
def crawl_command(
    url: Annotated[str, typer.Argument(help="Permitted start URL (robots.txt respected)")],
    out: Annotated[Path, typer.Option("--out")] = Path("pages.jsonl"),
    max_pages: Annotated[int, typer.Option("--max-pages", min=1)] = 200,
    concurrency: Annotated[int, typer.Option("--concurrency", min=1)] = 5,
    per_host: Annotated[int, typer.Option("--per-host", min=1)] = 2,
    host_delay: Annotated[float, typer.Option("--host-delay", min=0)] = 0.02,
    timeout: Annotated[float, typer.Option("--timeout", min=0.01)] = 5.0,
    log_path: Annotated[Path, typer.Option("--log-file")] = Path("crawl.log"),
) -> None:
    """Crawl allowed HTML pages into streaming JSONL with live Rich counters."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    crawl_logger = logging.getLogger("findex.crawl")
    crawl_logger.addHandler(handler)
    crawl_logger.setLevel(logging.INFO)
    state = CrawlStats()

    async def run() -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as stream:
            with Live(console=error_console, refresh_per_second=5, transient=True) as live:
                async for item in crawl(url, max_pages=max_pages, concurrency=concurrency,
                                        per_host=per_host, host_delay=host_delay,
                                        timeout_s=timeout, stats=state):
                    if 200 <= item.status < 300:
                        stream.write(json.dumps({"url": item.url, "text": item.text,
                                                 "status": item.status, "bytes": item.size_bytes,
                                                 "elapsed_s": item.elapsed_s}, ensure_ascii=False) + "\n")
                        stream.flush()
                    live.update(Panel(f"Pages: {state.pages}  Rate: {state.pages_per_second:.1f}/s  "
                                      f"In flight: {state.in_flight}  Errors: {state.errors}  "
                                      f"Queue: {state.queue_depth}", title="findex crawl"))

    started = monotonic()
    try:
        asyncio.run(run())
        out_console.print(f"Crawled {state.pages} pages in {monotonic() - started:.2f}s; "
                          f"errors={state.errors}; saved {out}; log {log_path}")
    except (OSError, ValueError) as error:
        fail(error)
    finally:
        crawl_logger.removeHandler(handler)
        handler.close()


@app.command("search")
def search_command(
    index: Annotated[Path, typer.Argument(help="Saved index file")],
    query: Annotated[str, typer.Argument(help="Boolean query, quotes for phrases")],
    scorer: Annotated[Literal["tfidf", "bm25"], typer.Option("--scorer")] = "bm25",
    top: Annotated[int, typer.Option("--top", min=1)] = 5,
    format: Annotated[Literal["pickle", "json"], typer.Option("--format")] = "pickle",
    corpus: Annotated[Path | None, typer.Option("--corpus", help="Folder for snippets")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable JSON to stdout")] = False,
    engine: Annotated[Literal["python", "numpy"], typer.Option("--engine")] = "numpy",
) -> None:
    """Rank matching documents and optionally emit clean JSON."""
    try:
        if not index.is_file():
            raise FileNotFoundError(f"Index file not found: {index}")
        with open_index(index, format) as opened:
            searcher = ranked_search_numpy if engine == "numpy" else ranked_search
            found = searcher(opened, query, BM25() if scorer == "bm25" else TfIdf(), top, corpus)
        if json_output:
            print(json.dumps([
                {"doc_id": item.doc_id, "title": item.title, "path": item.path,
                 "score": item.score, "snippet": item.snippet} for item in found
            ], ensure_ascii=False))
            return
        table = Table(title=f"Search: {query}")
        table.add_column("ID", justify="right")
        table.add_column("Document")
        table.add_column("Score", justify="right")
        table.add_column("Snippet")
        for item in found:
            table.add_row(str(item.doc_id), item.title, f"{item.score:.4f}", item.snippet)
        out_console.print(table)
        out_console.print(f"{len(found)} result(s) shown")
    except (OSError, UnicodeError, ValueError) as error:
        fail(error)


@app.command("stats")
def stats_command(data: Annotated[Path, typer.Argument(help="Folder with documents")]) -> None:
    """Print corpus statistics as a Rich table."""
    try:
        stats = streaming_stats(data)
        table = Table(title="Corpus statistics")
        table.add_column("Measure")
        table.add_column("Value", justify="right")
        for name, value in (("Documents", stats.documents), ("Tokens", stats.tokens), ("Vocabulary", stats.vocabulary)):
            table.add_row(name, str(value))
        out_console.print(table)
    except (OSError, UnicodeError, ValueError) as error:
        fail(error)


@app.command("serve")
def serve_command(
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8000,
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
    host: Annotated[str, typer.Option("--host")] = "0.0.0.0",
) -> None:
    """Serve the search UI and JSON API through Uvicorn."""
    import uvicorn

    uvicorn.run("findex.web.app:create_app", factory=True, host=host, port=int(os.getenv("PORT", port)), workers=workers)


@app.command("compact")
def compact_command(
    source: Annotated[Path, typer.Argument(help="Corpus directory")],
    out: Annotated[Path, typer.Option("--out")] = Path("postings_memmap"),
) -> None:
    """Persist postings as disk-backed NumPy int32 arrays."""
    try:
        save_memmap(build_index(source), out)
        out_console.print(f"NumPy memory-mapped postings saved to {out}")
    except (OSError, ValueError) as error:
        fail(error)


if __name__ == "__main__":
    app()
