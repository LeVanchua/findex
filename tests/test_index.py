import json
from pathlib import Path

import pytest

from findex.boolean import merge_and, merge_not, merge_or, search
from findex.inverted import DocMeta, Posting, build_index
from findex.serialization import load_index, save_index


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "a.txt").write_text("Python сервер сервер", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Python база", encoding="utf-8")
    (tmp_path / "c.txt").write_text("сервер база", encoding="utf-8")
    (tmp_path / "d.txt").write_text("Індекс", encoding="utf-8")
    return tmp_path


def test_index_structure_and_positions(corpus):
    index = build_index(corpus, with_positions=True)
    assert index.document_count == 4
    assert index.postings["сервер"] == [Posting(1, 2), Posting(3, 1)]
    assert index.documents[1] == DocMeta(1, "a.txt", "a", 3)
    assert index.positions["сервер"] == {1: [1, 2], 3: [0]}
    assert index.vocabulary_size == 4


@pytest.mark.parametrize("query,expected", [
    ("python", [1, 2]),
    ("python сервер", [1]),
    ("python AND сервер", [1]),
    ("python OR сервер", [1, 2, 3]),
    ("python NOT сервер", [2]),
    ("NOT сервер", [2, 4]),
    ("(python OR сервер) AND база", [2, 3]),
    ("python OR сервер AND база", [1, 2, 3]),
    ("(python OR база) NOT сервер", [2]),
    ("відсутнє", []),
])
def test_boolean_both_engines(corpus, query, expected):
    index = build_index(corpus)
    assert search(index, query, "merge") == expected
    assert search(index, query, "set") == expected


@pytest.mark.parametrize("query", ["", "python AND", "(python", "python OR OR база", "NOT"])
def test_bad_queries(corpus, query):
    with pytest.raises(ValueError):
        search(build_index(corpus), query)


def test_merge_algorithms():
    assert merge_and([1, 3, 5], [2, 3, 4]) == [3]
    assert merge_or([1, 3, 5], [2, 3, 4]) == [1, 2, 3, 4, 5]
    assert merge_not([1, 2, 3, 4], [2, 4]) == [1, 3]


@pytest.mark.parametrize("fmt", ["pickle", "json"])
def test_round_trip(corpus, tmp_path, fmt):
    index = build_index(corpus, with_positions=True)
    file = tmp_path / ("idx.bin" if fmt == "pickle" else "idx.json")
    save_index(index, file, fmt)
    loaded = load_index(file, fmt)
    assert loaded == index
    assert search(loaded, "python OR база") == [1, 2, 3]


def test_json_version_check(tmp_path):
    file = tmp_path / "bad.json"
    file.write_text(json.dumps({"version": 9}), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        load_index(file, "json")
