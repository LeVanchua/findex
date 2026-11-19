"""Installable findex CLI: Typer commands with Rich output and clean JSON."""

import json
import logging
import os
from pathlib import Path
import sys
from typing import Annotated, Literal

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import typer

from .parallel import ExecutorKind, build_parallel
from .model import open_index
from .ranking import BM25, TfIdf, ranked_search
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
            result, merge_seconds = build_parallel(data, workers=workers, executor=executor, with_positions=positions)
            save_index(result, out, format)
            progress.update(task, description="Index saved")
        out_console.print(f"Indexed {result.document_count} documents; {result.vocabulary_size} terms → {out}")
        logging.getLogger("findex").info("merge_seconds=%.6f executor=%s workers=%d", merge_seconds, executor, workers)
    except (OSError, UnicodeError, ValueError) as error:
        fail(error)


@app.command("search")
def search_command(
    index: Annotated[Path, typer.Argument(help="Saved index file")],
    query: Annotated[str, typer.Argument(help="Boolean query, quotes for phrases")],
    scorer: Annotated[Literal["tfidf", "bm25"], typer.Option("--scorer")] = "bm25",
    top: Annotated[int, typer.Option("--top", min=1)] = 5,
    format: Annotated[Literal["pickle", "json"], typer.Option("--format")] = "pickle",
    corpus: Annotated[Path | None, typer.Option("--corpus", help="Folder for snippets")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable JSON to stdout")] = False,
) -> None:
    """Rank matching documents and optionally emit clean JSON."""
    try:
        if not index.is_file():
            raise FileNotFoundError(f"Index file not found: {index}")
        with open_index(index, format) as opened:
            found = ranked_search(opened, query, BM25() if scorer == "bm25" else TfIdf(), top, corpus)
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


if __name__ == "__main__":
    app()
