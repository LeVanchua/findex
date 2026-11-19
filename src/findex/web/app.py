"""FastAPI boundary for the findex model; indexing stays outside the event loop."""

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from html import escape
import logging
from pathlib import Path
import re
import uuid
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.responses import Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

from findex.model import Index, load_model
from findex.ranking import BM25, TfIdf, ranked_search

log = logging.getLogger("findex.web")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
ScorerName = Literal["bm25", "tfidf"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")
    index_path: Path = Path("data/index.bin")
    corpus_path: Path | None = None
    index_format: Literal["pickle", "json"] = "pickle"


class SearchItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    doc_id: int
    title: str
    path: str
    score: float
    snippet: str


class SearchPage(BaseModel):
    q: str
    k: int
    scorer: ScorerName
    page: int
    total: int
    pages: int
    results: list[SearchItem]


class DocumentItem(BaseModel):
    doc_id: int
    title: str
    path: str
    length: int


class StatsItem(BaseModel):
    documents: int
    vocabulary: int
    avg_doc_length: float


class HealthItem(BaseModel):
    status: str


def get_index(request: Request) -> Index:
    index: Index | None = getattr(request.app.state, "index", None)
    if index is None:
        raise HTTPException(status_code=503, detail="Search index is not ready")
    return index


def get_settings(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


def do_search(index: Index, q: str, k: int, scorer: ScorerName, page: int, corpus: Path | None) -> SearchPage:
    # The query parser and BM25 are CPU work: execute this from a sync route/thread pool.
    total = len(index.cached_ids(q))
    hits = ranked_search(index, q, BM25() if scorer == "bm25" else TfIdf(), min(total, page * k), corpus)
    return SearchPage(
        q=q, k=k, scorer=scorer, page=page, total=total,
        pages=(total + k - 1) // k,
        results=[SearchItem.model_validate(hit) for hit in hits[(page - 1) * k:page * k]],
    )


def safe_highlight(snippet: str) -> str:
    """Escape untrusted document text before introducing a single <mark> element."""
    escaped = escape(snippet)
    return re.sub(r"\*\*(.+?)\*\*", r"<mark>\1</mark>", escaped)


def create_app(settings: Settings | None = None) -> FastAPI:
    configured = settings or Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
        application.state.index = None
        try:
            application.state.index = load_model(configured.index_path, configured.index_format)
            log.info("Loaded index %s", configured.index_path)
        except (OSError, ValueError) as error:
            log.error("Index not available at %s: %s", configured.index_path, error)
        try:
            yield
        finally:
            if application.state.index is not None:
                application.state.index.close()
                application.state.index = None

    application = FastAPI(title="findex search", version="0.7.0", lifespan=lifespan)
    application.state.settings = configured
    application.state.index = None

    @application.middleware("http")
    async def request_id_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))[:128]
        try:
            response: Response = await call_next(request)
        except Exception:
            log.exception("Unhandled error request_id=%s method=%s path=%s", request_id, request.method, request.url.path)
            response = JSONResponse(status_code=500, content={"detail": "Internal server error", "request_id": request_id})
        response.headers["X-Request-ID"] = request_id
        return response

    @application.get("/health", response_model=HealthItem)
    def health(request: Request) -> HealthItem:
        if request.app.state.index is None:
            raise HTTPException(status_code=503, detail="Search index is not ready")
        return HealthItem(status="ok")

    @application.get("/stats", response_model=StatsItem)
    def stats(index: Annotated[Index, Depends(get_index)]) -> StatsItem:
        return StatsItem(documents=index.num_docs, vocabulary=len(index), avg_doc_length=index.avg_doc_length)

    @application.get("/docs/{doc_id}", response_model=DocumentItem)
    def document(doc_id: int, index: Annotated[Index, Depends(get_index)]) -> DocumentItem:
        meta = index.documents.get(doc_id)
        if meta is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return DocumentItem(doc_id=meta.doc_id, title=meta.title, path=meta.path, length=meta.length)

    @application.get("/document/{doc_id}", response_class=HTMLResponse)
    def document_page(
        doc_id: int,
        request: Request,
        index: Annotated[Index, Depends(get_index)],
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> HTMLResponse:
        meta = index.documents.get(doc_id)
        if meta is None:
            raise HTTPException(status_code=404, detail="Document not found")
        body = "Текст документа недоступний у цьому середовищі."
        if settings.corpus_path is not None:
            base = settings.corpus_path.resolve()
            file_path = (base / meta.path).resolve()
            if file_path.is_relative_to(base) and file_path.is_file():
                body = file_path.read_text(encoding="utf-8")[:100_000]
        return templates.TemplateResponse(request, "document.html", {"meta": meta, "body": body})

    def search_args(
        q: Annotated[str, Query(min_length=1, max_length=400)],
        k: Annotated[int, Query(ge=1, le=100)] = 10,
        scorer: ScorerName = "bm25",
        page: Annotated[int, Query(ge=1)] = 1,
    ) -> tuple[str, int, ScorerName, int]:
        return q, k, scorer, page

    @application.get("/search", response_model=SearchPage)
    def search(
        args: Annotated[tuple[str, int, ScorerName, int], Depends(search_args)],
        index: Annotated[Index, Depends(get_index)],
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> SearchPage:
        try:
            return do_search(index, *args, corpus=settings.corpus_path)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.get("/bench/async-search", response_model=SearchPage, include_in_schema=False)
    async def async_search_benchmark(
        args: Annotated[tuple[str, int, ScorerName, int], Depends(search_args)],
        index: Annotated[Index, Depends(get_index)],
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> SearchPage:
        """Intentional blocking control case for measured def-vs-async def comparison."""
        return do_search(index, *args, corpus=settings.corpus_path)

    @application.get("/", response_class=HTMLResponse)
    def homepage(
        request: Request,
        index: Annotated[Index, Depends(get_index)],
        settings: Annotated[Settings, Depends(get_settings)],
        q: str = "",
        page: Annotated[int, Query(ge=1)] = 1,
    ) -> HTMLResponse:
        results: SearchPage | None = None
        error: str | None = None
        if q.strip():
            try:
                results = do_search(index, q.strip(), 10, "bm25", page, settings.corpus_path)
            except ValueError as exc:
                error = str(exc)
        return templates.TemplateResponse(request, "index.html", {
            "q": q, "results": results, "error": error, "safe_highlight": safe_highlight,
            "documents": index.num_docs, "vocabulary": len(index),
        })

    return application


app = create_app()
