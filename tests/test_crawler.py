"""Offline crawl tests: no external requests are made."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from findex.app import app
from findex.corpus import iter_documents
from findex.crawler import CrawlStats, FetchResult, RobotsCache, canonical_url, crawl, fetch
from findex.inverted import build_index


@pytest.mark.parametrize("source,expected", [
    ("HTTPS://EXAMPLE.COM:443/a/#part", "https://example.com/a"),
    ("https://Example.com/?b=2&a=1#x", "https://example.com/?a=1&b=2"),
    ("http://EXAMPLE.com:80/", "http://example.com/"),
])
def test_canonical_url(source: str, expected: str) -> None:
    assert canonical_url(source) == expected


def test_invalid_url() -> None:
    with pytest.raises(ValueError):
        canonical_url("file:///tmp/file")


def test_fetch_retry_after_and_404() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if request.url.path == "/missing":
            return httpx.Response(404)
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, headers={"Content-Type": "text/html"},
                              text="<h1>Hello</h1><script>hidden</script><a href='/next'>Next</a>")

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch(client, "https://demo.test/", backoff_base=0)
            assert result.status == 200 and result.attempts == 2
            assert "Hello" in result.text and "hidden" not in result.text
            assert result.links == ("/next",)
            missing = await fetch(client, "https://demo.test/missing", backoff_base=0)
            assert missing.status == 404 and missing.attempts == 1
    asyncio.run(scenario())


def test_fetch_5xx_and_timeout() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if request.url.path == "/timeout":
            raise httpx.ReadTimeout("slow")
        return httpx.Response(503)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            failed = await fetch(client, "https://demo.test/fail", max_attempts=3, backoff_base=0)
            assert failed.status == 503 and failed.attempts == 3
            timed = await fetch(client, "https://demo.test/timeout", max_attempts=3, backoff_base=0)
            assert timed.status == 0 and timed.attempts == 3
            assert timed.error == "ReadTimeout"
    asyncio.run(scenario())
    assert calls == 6


def test_robots_cache_per_host_and_disallow() -> None:
    robots_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal robots_calls
        if request.url.path == "/robots.txt":
            robots_calls += 1
            return httpx.Response(200, text="User-agent: *\nDisallow: /secret")
        return httpx.Response(200)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            cache = RobotsCache()
            assert await cache.allowed(client, "https://one.test/ok", "findex-lab06")
            assert not await cache.allowed(client, "https://one.test/secret", "findex-lab06")
            assert await cache.allowed(client, "https://two.test/ok", "findex-lab06")
    asyncio.run(scenario())
    assert robots_calls == 2


def test_crawl_bounded_dedup_and_jsonl_index(tmp_path: Path) -> None:
    fetched: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /")
        fetched.append(str(request.url))
        links = "".join(f'<a href="/p/{i}?b=2&a=1#fragment">page {i}</a>' for i in range(25))
        return httpx.Response(200, headers={"Content-Type": "text/html"},
                              text=f"<title>Page</title> Python searchable text {links}")

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            stats = CrawlStats()
            pages = []
            async for result in crawl("https://demo.test/", max_pages=20, concurrency=5,
                                      per_host=2, host_delay=0, client=client, stats=stats):
                pages.append(result)
            assert len(pages) == 20 and stats.pages == 20
            assert len(fetched) == 20 and len(set(fetched)) == 20
            with (tmp_path / "pages.jsonl").open("w", encoding="utf-8") as stream:
                for page in pages:
                    stream.write(json.dumps({"url": page.url, "text": page.text}) + "\n")
    asyncio.run(scenario())
    documents = list(iter_documents(tmp_path))
    assert len(documents) == 20
    index = build_index(tmp_path)
    assert index.document_count == 20
    assert len(index.postings["python"]) == 20


def test_robots_disallowed_page_never_fetched() -> None:
    visited: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        visited.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /")
        return httpx.Response(200, text="<h1>Forbidden</h1>")

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            pages = [page async for page in crawl("https://demo.test/", client=client,
                                                  max_pages=5, host_delay=0)]
            assert len(pages) == 1 and pages[0].error == "robots.txt disallow"
    asyncio.run(scenario())
    assert visited == ["/robots.txt"]


def test_cli_crawl_streaming(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_crawl(*args: object, **kwargs: object):
        yield FetchResult("https://demo.test/", 200, 42, 0.01, text="hello python")

    monkeypatch.setattr("findex.app.crawl", fake_crawl)
    output = tmp_path / "pages.jsonl"
    log = tmp_path / "crawl.log"
    result = CliRunner().invoke(app, ["crawl", "https://demo.test/", "--max-pages", "1",
                                      "--out", str(output), "--log-file", str(log)])
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text(encoding="utf-8"))["text"] == "hello python"
    assert log.exists()
