from pathlib import Path

import pytest

from findex.corpus import iter_documents
from findex.stats import greedy_stats, streaming_stats


def test_iter_documents_sorted_and_txt_only(tmp_path: Path):
    (tmp_path / "b.txt").write_text("два", encoding="utf-8")
    (tmp_path / "a.txt").write_text("один", encoding="utf-8")
    (tmp_path / "ignore.md").write_text("skip", encoding="utf-8")
    docs = iter_documents(tmp_path)
    assert iter(docs) is docs
    assert [(d.path.name, d.text) for d in docs] == [("a.txt", "один"), ("b.txt", "два")]


def test_stats_match(tmp_path: Path):
    (tmp_path / "1.txt").write_text("Слово слово Python3", encoding="utf-8")
    (tmp_path / "2.txt").write_text("слово П'ять", encoding="utf-8")
    streaming = streaming_stats(tmp_path)
    greedy = greedy_stats(tmp_path)
    assert streaming.documents == greedy.documents == 2
    assert streaming.tokens == greedy.tokens == 5
    assert streaming.vocabulary == greedy.vocabulary == 3
    assert streaming.term_counts == greedy.term_counts
    assert streaming.top_terms() == [("слово", 3), ("python3", 1), ("п'ять", 1)]


def test_missing_corpus_raises(tmp_path: Path):
    with pytest.raises(NotADirectoryError):
        list(iter_documents(tmp_path / "missing"))
