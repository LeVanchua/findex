# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY README.md ./README.md
RUN uv sync --frozen --no-dev
# An index must exist in the image: Git intentionally ignores user-generated data.
# This reproducible synthetic dataset powers the public teaching demo.
RUN .venv/bin/python -c "from findex.sample_data import generate_corpus; from findex.inverted import build_index; from findex.serialization import save_index; generate_corpus('data/corpus', 1000, 5); save_index(build_index('data/corpus'), 'data/index.bin', 'pickle')"

FROM python:3.13-slim AS runtime
WORKDIR /app
RUN groupadd --system findex && useradd --system --gid findex --home-dir /app findex
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/data /app/data
COPY --from=builder /app/src /app/src
ENV PATH="/app/.venv/bin:$PATH" INDEX_PATH=/app/data/index.bin CORPUS_PATH=/app/data/corpus PYTHONUNBUFFERED=1 PORT=8000
USER findex
EXPOSE 8000
CMD ["sh", "-c", "exec findex serve --port ${PORT:-8000} --workers ${WEB_CONCURRENCY:-1}"]
