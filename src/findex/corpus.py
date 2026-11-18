"""Read documents lazily, one file at a time."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


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
