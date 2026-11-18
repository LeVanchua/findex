import json
from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import given, settings, strategies as st
import pytest
from typer.testing import CliRunner

from findex.app import app
from findex.boolean import merge_and, merge_not, merge_or
from findex.inverted import build_index
from findex.serialization import load_index, save_index
from findex.tokenizer import tokenize


runner = CliRunner()


@pytest.mark.parametrize("source, expected", [
    ("ＰＹＴＨＯＮ", ["python"]),
    ("П’ять", ["п'ять"]),
    ("онлайн-пошук", ["онлайн", "пошук"]),
    ("python3 2026", ["python3", "2026"]),
    ("НАВЧАННЯ", ["навчання"]),
    ("a_b", ["a", "b"]),
])
def test_unicode_parametrized(source, expected):
    assert list(tokenize(source)) == expected


def test_cli_help():
    output = runner.invoke(app, ["--help"])
    assert output.exit_code == 0
    for command in ("index", "search", "stats"):
        assert command in output.stdout


def test_cli_end_to_end(tiny_corpus: Path, tmp_path: Path):
    file = tmp_path / "saved.bin"
    created = runner.invoke(app, ["index", str(tiny_corpus), "--out", str(file), "--positions"])
    assert created.exit_code == 0, created.output
    assert file.is_file()
    result = runner.invoke(app, ["search", str(file), "python", "--json", "--top", "2"])
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)
    assert len(parsed) == 2
    assert {row["doc_id"] for row in parsed} == {1, 2}
    stats = runner.invoke(app, ["stats", str(tiny_corpus)])
    assert stats.exit_code == 0
    assert "Documents" in stats.stdout


def test_cli_missing_index(tmp_path: Path):
    result = runner.invoke(app, ["search", str(tmp_path / "absent.bin"), "python"])
    assert result.exit_code != 0
    assert "Index file not found" in result.output
    assert "Traceback" not in result.output


def test_cli_broken_query(tiny_corpus: Path, tmp_path: Path):
    file = tmp_path / "index.bin"
    save_index(build_index(tiny_corpus), file)
    result = runner.invoke(app, ["search", str(file), "python AND", "--json"])
    assert result.exit_code != 0
    assert "Error:" in result.output
    assert "Traceback" not in result.output


@given(st.lists(st.text(alphabet="абвгд", min_size=1, max_size=8), max_size=12))
@settings(max_examples=30)
def test_property_save_load_roundtrip(words):
    with TemporaryDirectory() as folder:
        directory = Path(folder)
        corpus = directory / "corpus"
        corpus.mkdir()
        (corpus / "a.txt").write_text(" ".join(words), encoding="utf-8")
        index = build_index(corpus)
        target = directory / "index.json"
        save_index(index, target, "json")
        assert load_index(target, "json") == index


@given(st.lists(st.lists(st.sampled_from(["python", "сервер", "текст"]), max_size=15), min_size=1, max_size=10))
@settings(max_examples=30)
def test_property_sorted_postings(documents):
    with TemporaryDirectory() as folder:
        corpus = Path(folder)
        for n, tokens in enumerate(documents):
            (corpus / f"doc{n:03d}.txt").write_text(" ".join(tokens), encoding="utf-8")
        index = build_index(corpus)
        for values in index.postings.values():
            ids = [p.doc_id for p in values]
            assert ids == sorted(set(ids))


@given(st.sets(st.integers(min_value=0, max_value=200)), st.sets(st.integers(min_value=0, max_value=200)))
@settings(max_examples=100)
def test_property_merge_matches_sets(left, right):
    a, b = sorted(left), sorted(right)
    assert merge_and(a, b) == sorted(left & right)
    assert merge_or(a, b) == sorted(left | right)
    assert merge_not(a, b) == sorted(left - right)
