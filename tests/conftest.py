from pathlib import Path

import pytest


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> Path:
    (tmp_path / "a.txt").write_text("Python Unicode Python", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Python сервер", encoding="utf-8")
    (tmp_path / "c.txt").write_text("сервер пошук", encoding="utf-8")
    return tmp_path
