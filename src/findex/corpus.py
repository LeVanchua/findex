"""Read documents lazily, one file at a time."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator, cast


@dataclass(frozen=True, slots=True)
class Document:
    path: Path
    text: str


def iter_documents(root: Path | str) -> Iterator[Document]:
    """Yield UTF-8 .txt documents in stable path order.

    The full file list is not materialized. One document's contents is kept
    in memory at a time; an individual huge file may still be large.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Corpus directory does not exist: {root}")
    # os.walk does not preload every path; sort within each directory only.
    import os

    for directory, dirs, files in os.walk(root):
        dirs.sort()
        for filename in sorted(files):
            path = Path(directory) / filename
            if path.suffix.lower() == ".txt":
                yield Document(path=path, text=path.read_text(encoding="utf-8"))
            elif path.suffix.lower() == ".jsonl":
                # Stream one web page per line, without materializing the crawl.
                with path.open("r", encoding="utf-8") as stream:
                    for line_number, line in enumerate(stream, 1):
                        raw: object = json.loads(line)
                        if not isinstance(raw, dict):
                            raise ValueError(f"Invalid crawl record at {path}:{line_number}")
                        record = cast(dict[str, object], raw)
                        page_text = record.get("text")
                        if not isinstance(page_text, str):
                            raise ValueError(f"Invalid crawl text at {path}:{line_number}")
                        virtual = path.parent / f"{path.stem}_page_{line_number:06d}.txt"
                        yield Document(path=virtual, text=page_text)
