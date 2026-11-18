"""Round-trip indexing via pickle and JSON.

SECURITY: Never unpickle untrusted files: pickle.load can execute code.
JSON stores only primitive data, validated before model reconstruction.
"""

import json
import pickle
from pathlib import Path

from .inverted import DocMeta, InvertedIndex, Posting


def save_index(index: InvertedIndex, path: str | Path, format: str = "pickle") -> None:
    path = Path(path)
    if format == "pickle":
        with path.open("wb") as stream:
            pickle.dump(index, stream, protocol=pickle.HIGHEST_PROTOCOL)
    elif format == "json":
        data = {
            "version": 1,
            "postings": {t: [[p.doc_id, p.frequency] for p in values] for t, values in index.postings.items()},
            "documents": {str(k): [m.doc_id, m.path, m.title, m.length] for k, m in index.documents.items()},
            "positions": index.positions,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    else:
        raise ValueError("Format must be pickle or json")


def load_index(path: str | Path, format: str = "pickle") -> InvertedIndex:
    path = Path(path)
    if format == "pickle":
        # Caller must only pass a file created by a trusted source.
        with path.open("rb") as stream:
            index = pickle.load(stream)  # noqa: S301 - explicitly trusted input only
        if not isinstance(index, InvertedIndex):
            raise ValueError("Invalid pickle index")
        return index
    if format == "json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("version") != 1:
            raise ValueError("Unsupported index version")
        postings = {term: [Posting(int(doc), int(freq)) for doc, freq in records]
                    for term, records in raw["postings"].items()}
        documents = {int(key): DocMeta(int(row[0]), str(row[1]), str(row[2]), int(row[3]))
                     for key, row in raw["documents"].items()}
        positions = (None if raw["positions"] is None else {
            term: {int(k): list(map(int, offsets)) for k, offsets in docs.items()}
            for term, docs in raw["positions"].items()
        })
        return InvertedIndex(postings, documents, positions)
    raise ValueError("Format must be pickle or json")
