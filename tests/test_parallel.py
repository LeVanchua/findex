"""Lab 05 correctness and worker-failure tests."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.app import app
from findex.inverted import build_index
from findex.parallel import build_parallel, build_partial, corpus_paths, merge


@pytest.mark.parametrize("positions", [False, True])
def test_serial_matches_previous_lab(tiny_corpus: Path, positions: bool) -> None:
    expected = build_index(tiny_corpus, with_positions=positions)
    paths = corpus_paths(tiny_corpus)
    assert merge([build_partial(tiny_corpus, paths, 1, positions)]) == expected
    assert build_parallel(tiny_corpus, 1, "serial", positions)[0] == expected


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
def test_all_executors_identical(tiny_corpus: Path, workers: int, executor: str) -> None:
    from typing import cast
    from findex.parallel import ExecutorKind

    expected = build_index(tiny_corpus, with_positions=True)
    result, seconds = build_parallel(tiny_corpus, workers, cast(ExecutorKind, executor), True)
    assert result == expected
    assert seconds >= 0


def test_duplicate_doc_ids_rejected(tiny_corpus: Path) -> None:
    paths = corpus_paths(tiny_corpus)
    shard = build_partial(tiny_corpus, paths[:1], 1)
    with pytest.raises(ValueError, match="Duplicate"):
        merge([shard, shard])


def test_worker_exception_propagates(tiny_corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("findex.parallel.corpus_paths", lambda root: ["missing.txt"])
    with pytest.raises(FileNotFoundError):
        build_parallel(tiny_corpus, workers=2, executor="processes")


def test_cli_parallel(tiny_corpus: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["index", str(tiny_corpus), "--out", str(tmp_path / "index.bin"),
                                       "--workers", "2", "--executor", "processes"])
    assert result.exit_code == 0, result.output
