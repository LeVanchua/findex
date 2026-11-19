"""Polite async HTTP crawler with bounded scheduling and streaming results."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
import logging
import random
from time import monotonic, perf_counter
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx

USER_AGENT = "findex-lab06/0.6 (+https://github.com/LeVanchua/findex/issues)"
logger = logging.getLogger("findex.crawl")


def canonical_url(url: str) -> str:
    """Normalize scheme/host, query ordering, fragment and trailing slash."""
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    if scheme not in ("http", "https") or not host:
        raise ValueError(f"Unsupported URL: {url}")
    port = parsed.port
    authority = f"[{host}]" if ":" in host else host
    if port is not None and (scheme, port) not in (("http", 80), ("https", 443)):
        authority += f":{port}"
    path = parsed.path or "/"
    path = path.rstrip("/") or "/"
    query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
    return urlunsplit((scheme, authority, path, query, ""))


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "noscript"):
            self.hidden += 1
        if tag == "a":
            for key, value in attrs:
                if key == "href" and value:
                    self.links.append(value)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript"):
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data: str) -> None:
        if self.hidden == 0 and data.strip():
            self.parts.append(data.strip())


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    status: int
    size_bytes: int
    elapsed_s: float
    text: str = ""
    links: tuple[str, ...] = ()
    error: str = ""
    attempts: int = 1


def retry_delay(response: httpx.Response | None, attempt: int, base: float = 0.1) -> float:
    if response is not None:
        value = response.headers.get("retry-after", "")
        if value:
            try:
                return max(0.0, min(30.0, float(value)))
            except ValueError:
                try:
                    when = parsedate_to_datetime(value)
                    return min(30.0, max(0.0, (when - datetime.now(timezone.utc)).total_seconds()))
                except (TypeError, ValueError, OverflowError):
                    pass
    return min(10.0, base * (2 ** (attempt - 1)) + random.uniform(0, base))


async def fetch(client: httpx.AsyncClient, url: str, timeout_s: float = 5.0,
                max_attempts: int = 3, backoff_base: float = 0.1) -> FetchResult:
    """Retry 429, 5xx and timeouts (not other 4xx); never block the event loop."""
    started = perf_counter()
    for attempt in range(1, max_attempts + 1):
        response: httpx.Response | None = None
        try:
            async with asyncio.timeout(timeout_s):
                response = await client.get(url)
            size = len(response.content)
            status = response.status_code
            if status == 429 or 500 <= status <= 599:
                logger.warning("retry url=%s status=%d attempt=%d", url, status, attempt)
                if attempt < max_attempts:
                    await asyncio.sleep(retry_delay(response, attempt, backoff_base))
                    continue
            if 200 <= status < 300 and "html" in response.headers.get("content-type", "text/html"):
                parser = PageParser()
                parser.feed(response.text)
                return FetchResult(url, status, size, perf_counter() - started,
                                   " ".join(parser.parts), tuple(parser.links), attempts=attempt)
            return FetchResult(url, status, size, perf_counter() - started,
                               error=f"HTTP {status}" if status >= 400 else "", attempts=attempt)
        except (TimeoutError, httpx.TimeoutException) as error:
            logger.warning("retry url=%s timeout attempt=%d", url, attempt)
            if attempt == max_attempts:
                return FetchResult(url, 0, 0, perf_counter() - started, error=type(error).__name__, attempts=attempt)
            await asyncio.sleep(retry_delay(None, attempt, backoff_base))
        except httpx.HTTPError as error:
            return FetchResult(url, 0, 0, perf_counter() - started, error=type(error).__name__, attempts=attempt)
    raise RuntimeError("unreachable")


class RobotsCache:
    """Cache one robots.txt per origin, with a lock for concurrent first requests."""

    def __init__(self) -> None:
        self._robots: dict[str, RobotFileParser] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def allowed(self, client: httpx.AsyncClient, url: str, user_agent: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        async with self._locks.setdefault(origin, asyncio.Lock()):
            if origin not in self._robots:
                parser = RobotFileParser()
                try:
                    async with asyncio.timeout(5.0):
                        response = await client.get(f"{origin}/robots.txt")
                    if response.status_code == 404:
                        parser.parse(["User-agent: *", "Allow: /"])
                    elif response.status_code == 200:
                        parser.parse(response.text.splitlines())
                    else:
                        parser.parse(["User-agent: *", "Disallow: /"])
                except (httpx.HTTPError, TimeoutError):
                    parser.parse(["User-agent: *", "Disallow: /"])
                self._robots[origin] = parser
        return self._robots[origin].can_fetch(user_agent, url)


@dataclass(slots=True)
class CrawlStats:
    pages: int = 0
    errors: int = 0
    in_flight: int = 0
    queue_depth: int = 0
    started: float = field(default_factory=monotonic)

    @property
    def pages_per_second(self) -> float:
        return self.pages / max(0.000001, monotonic() - self.started)


async def crawl(start_url: str, *, max_pages: int = 200, concurrency: int = 5,
                per_host: int = 2, host_delay: float = 0.02, timeout_s: float = 5.0,
                stats: CrawlStats | None = None, client: httpx.AsyncClient | None = None,
                user_agent: str = USER_AGENT) -> AsyncIterator[FetchResult]:
    """Bounded async generator. Owns its client unless supplied by a test."""
    if max_pages < 1 or concurrency < 1 or per_host < 1 or host_delay < 0:
        raise ValueError("max_pages/concurrency/per_host must be positive; host_delay nonnegative")
    seed = canonical_url(start_url)
    seed_origin = urlsplit(seed).netloc
    state = stats if stats is not None else CrawlStats()
    seen: set[str] = {seed}
    pending: asyncio.Queue[str] = asyncio.Queue()
    pending.put_nowait(seed)
    results: asyncio.Queue[FetchResult | None] = asyncio.Queue()
    global_sem = asyncio.Semaphore(concurrency)
    host_sems: dict[str, asyncio.Semaphore] = {}
    host_locks: dict[str, asyncio.Lock] = {}
    last_request: dict[str, float] = {}
    robots = RobotsCache()
    owned = client is None
    http = client if client is not None else httpx.AsyncClient(headers={"User-Agent": user_agent},
                                                               follow_redirects=False, trust_env=False)

    async def worker() -> None:
        while True:
            url = await pending.get()
            state.queue_depth = pending.qsize()
            try:
                origin = urlsplit(url).netloc
                if not await robots.allowed(http, url, user_agent):
                    outcome = FetchResult(url, 0, 0, 0.0, error="robots.txt disallow")
                else:
                    sem = host_sems.setdefault(origin, asyncio.Semaphore(per_host))
                    async with global_sem, sem:
                        async with host_locks.setdefault(origin, asyncio.Lock()):
                            pause = last_request.get(origin, 0.0) + host_delay - monotonic()
                            if pause > 0:
                                await asyncio.sleep(pause)
                            last_request[origin] = monotonic()
                        state.in_flight += 1
                        try:
                            outcome = await fetch(http, url, timeout_s=timeout_s)
                        finally:
                            state.in_flight -= 1
                state.pages += int(200 <= outcome.status < 300)
                state.errors += int(bool(outcome.error))
                logger.info("url=%s status=%s bytes=%d elapsed=%.4f attempts=%d error=%s",
                            url, outcome.status, outcome.size_bytes, outcome.elapsed_s,
                            outcome.attempts, outcome.error or "none")
                for href in outcome.links:
                    try:
                        destination = canonical_url(urljoin(url, href))
                    except ValueError:
                        continue
                    if (urlsplit(destination).netloc == seed_origin and destination not in seen
                            and len(seen) < max_pages):
                        seen.add(destination)
                        pending.put_nowait(destination)
                state.queue_depth = pending.qsize()
                await results.put(outcome)
            finally:
                pending.task_done()

    async def finish(tasks: list[asyncio.Task[None]]) -> None:
        await pending.join()
        for task in tasks:
            task.cancel()
        await results.put(None)

    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(worker()) for _ in range(concurrency)]
            finisher = group.create_task(finish(tasks))
            try:
                while True:
                    item = await results.get()
                    if item is None:
                        break
                    yield item
            finally:
                for task in tasks:
                    task.cancel()
                finisher.cancel()
    finally:
        if owned:
            await http.aclose()
