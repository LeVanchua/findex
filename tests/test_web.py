"""FastAPI boundary tests: synthetic in-memory index, never a network or index file."""

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from findex.inverted import DocMeta, InvertedIndex, Posting
from findex.model import Index
from findex.web.app import Settings, create_app, get_index


@pytest.fixture
def index() -> Index:
    return Index(InvertedIndex(
        postings={"python": [Posting(1, 2), Posting(2, 1)], "search": [Posting(1, 1)]},
        documents={1: DocMeta(1, "python.txt", "Python Search", 3), 2: DocMeta(2, "other.txt", "Python Notes", 1)},
    ))


@pytest.fixture
def client(index: Index) -> TestClient:
    app = create_app(Settings(index_path=Path("/nonexistent/findex-test-index.bin")))
    app.dependency_overrides[get_index] = lambda: index
    return TestClient(app)


def test_search_and_pagination(client: TestClient) -> None:
    first = client.get("/search", params={"q": "python", "k": 1, "page": 1})
    assert first.status_code == 200
    assert first.json()["total"] == 2
    assert first.json()["pages"] == 2
    assert len(first.json()["results"]) == 1
    second = client.get("/search", params={"q": "python", "k": 1, "page": 2})
    assert second.status_code == 200
    assert first.json()["results"][0]["doc_id"] != second.json()["results"][0]["doc_id"]


@pytest.mark.parametrize("params", [
    {"q": ""}, {"q": "python", "k": 0}, {"q": "python", "page": 0},
    {"q": "python", "scorer": "unknown"}, {"q": "python", "k": 101},
])
def test_invalid_query_422(client: TestClient, params: dict[str, str | int]) -> None:
    assert client.get("/search", params=params).status_code == 422


def test_documents(client: TestClient) -> None:
    assert client.get("/docs/1").json()["title"] == "Python Search"
    assert client.get("/docs/404").status_code == 404


def test_stats_and_browser(client: TestClient) -> None:
    assert client.get("/stats").json()["documents"] == 2
    response = client.get("/", params={"q": "python"})
    assert response.status_code == 200
    assert "Python Search" in response.text


def test_unready_health() -> None:
    app = create_app(Settings(index_path=Path("/nonexistent/findex-test-index.bin")))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 503
        assert client.get("/search", params={"q": "python"}).status_code == 503


def test_unexpected_error_has_request_id_without_traceback(index: Index) -> None:
    app = create_app(Settings(index_path=Path("/nonexistent/findex-test-index.bin")))

    def broken() -> Index:
        raise RuntimeError("private implementation traceback")

    app.dependency_overrides[get_index] = broken
    client = TestClient(app, raise_server_exceptions=False)
    result = client.get("/search", params={"q": "python"}, headers={"X-Request-ID": "test-123"})
    assert result.status_code == 500
    assert result.json()["request_id"] == "test-123"
    assert "private implementation" not in result.text


def test_no_raw_html_in_snippet() -> None:
    from findex.web.app import safe_highlight

    assert safe_highlight("<img src=x> **Python**") == "&lt;img src=x&gt; <mark>Python</mark>"
