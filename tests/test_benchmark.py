"""pytest-benchmark regressions, executed in GitHub Actions CI."""

from pathlib import Path

from pytest_benchmark.fixture import BenchmarkFixture

from findex.inverted import build_index
from findex.model import Index
from findex.numpy_rank import ranked_search_numpy
from findex.ranking import BM25
from findex.sample_data import generate_corpus


def test_benchmark_search(benchmark: BenchmarkFixture, tmp_path: Path) -> None:
    generate_corpus(tmp_path, 100, 1)
    index = Index(build_index(tmp_path))
    index.numpy_index  # warm up immutable arrays
    result = benchmark(ranked_search_numpy, index, "python OR база OR тестування", BM25(), 10)
    assert result


def test_benchmark_build_index(benchmark: BenchmarkFixture, tmp_path: Path) -> None:
    generate_corpus(tmp_path, 50, 1)
    result = benchmark(build_index, tmp_path)
    assert result.document_count == 50
