from pathlib import Path
import logging

import pytest

from findex.inverted import build_index
from findex.model import Index, open_index
from findex.query import And, Not, Or, Phrase, Term, parse_query
from findex.ranking import BM25, TfIdf, make_snippet, ranked_search
from findex.serialization import save_index


@pytest.fixture
def files(tmp_path):
    (tmp_path / "a.txt").write_text("Python event loop async async", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Python java event loop", encoding="utf-8")
    (tmp_path / "c.txt").write_text("Python await event loop", encoding="utf-8")
    (tmp_path / "d.txt").write_text("Java server", encoding="utf-8")
    return tmp_path


def test_mapping_protocol(files):
    index = Index(build_index(files, with_positions=True))
    assert len(index) == 7
    assert "python" in index
    assert index["python"][0].doc_id == 1
    assert "python" in list(iter(index))
    assert index.num_docs == 4
    assert index.avg_doc_length == pytest.approx(3.75)
    assert index.df("event") == 3
    assert "Index(" in repr(index)


def test_parser_tree_not_strings():
    expected = And(And(And(Term("python"), Or(Term("async"), Term("await"))),
                       Not(Term("java"))), Phrase(("event", "loop")))
    assert parse_query('python AND (async OR await) NOT java "event loop"') == expected
    assert parse_query("a OR b AND NOT c") == Or(Term("a"), And(Term("b"), Not(Term("c"))))
    assert (Term("a") & ~Term("b")) == And(Term("a"), Not(Term("b")))
    assert (Term("a") | Term("b")) == Or(Term("a"), Term("b"))


def test_phrase_and_boolean(files):
    index = Index(build_index(files, with_positions=True))
    assert parse_query('"event loop"').evaluate(index) == {1, 2, 3}
    assert parse_query('python AND (async OR await) NOT java "event loop"').evaluate(index) == {1, 3}
    with pytest.raises(ValueError, match="positions"):
        parse_query('"event loop"').evaluate(Index(build_index(files)))


@pytest.mark.parametrize("query", ["", "a AND", "(a", '"unterminated', '""', "a OR OR b"])
def test_parser_errors(query):
    with pytest.raises(ValueError):
        parse_query(query)


def test_scorer_sanity():
    tfidf = TfIdf()
    bm25 = BM25()
    assert tfidf.score(1, 1, 10, 100, 100) > tfidf.score(1, 50, 10, 100, 100)
    assert bm25.score(1, 1, 10, 100, 100) > bm25.score(1, 50, 10, 100, 100)
    increment_20 = bm25.score(20, 2, 100, 100, 100) - bm25.score(19, 2, 100, 100, 100)
    increment_1 = bm25.score(1, 2, 100, 100, 100)
    assert increment_20 / increment_1 < 0.01
    assert bm25.score(1, 2, 20, 100, 100) > bm25.score(1, 2, 1000, 100, 100)


def test_rank_cache_and_context(files, tmp_path, caplog):
    output = tmp_path / "index.bin"
    save_index(build_index(files, with_positions=True), output)
    with open_index(output) as index:
        with caplog.at_level(logging.INFO):
            first = ranked_search(index, '"event loop" AND python NOT java', BM25(), top_k=3, corpus=files)
            again = ranked_search(index, '"event loop" AND python NOT java', BM25(), top_k=3, corpus=files)
        assert [r.doc_id for r in first] == [3, 1]
        assert first == again
        assert all(r.snippet for r in first)
        assert index.cached_ids.cache_info().hits >= 1
        assert "cache MISS" in caplog.text and "cache HIT" in caplog.text
        assert "@timed" in caplog.text
    with pytest.raises(RuntimeError, match="closed"):
        index.df("python")


def test_context_closes_after_error(files, tmp_path):
    output = tmp_path / "saved.bin"
    save_index(build_index(files), output)
    ref = None
    with pytest.raises(ZeroDivisionError):
        with open_index(output) as idx:
            ref = idx
            1 / 0
    assert repr(ref) == "Index(closed=True)"


def test_snippet_highlight():
    snippet = make_snippet("the Python event loop works", ("python", "loop"), radius=5)
    assert "**Python**" in snippet
    assert "**loop**" not in snippet  # outside ±5 window
