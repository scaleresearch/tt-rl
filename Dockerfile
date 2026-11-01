# Host test environment: Linux x86_64 and Python 3.12, no chip needed.
#
#   docker build --platform linux/amd64 -t tt-rl-test .
#   docker run --rm tt-rl-test
FROM ubuntu:24.04

# libatomic1 is needed by ttnn, psmisc (fuser) by chip liveness checks, and
# git by the provenance tests.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates git libatomic1 psmisc \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_PYTHON=3.12

WORKDIR /src
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra dev --no-install-project
COPY . .
RUN uv sync --frozen --extra dev

CMD ["uv", "run", "pytest", "-m", "not tt"]
