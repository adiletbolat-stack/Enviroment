# Reproducible environment for: ai-companion-markers
# CPU-only (no CUDA needed). Python and OS are pinned via the base image tag.
FROM python:3.11.9-slim-bookworm

# Deterministic, quiet Python behaviour
ENV PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=UTC

# Minimal system dependencies (git is needed by nbstripout and for the commit workflow)
RUN apt-get update \
    && apt-get install -y --no-install-recommends git make \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (avoids root-owned files in the mounted repository)
RUN useradd --create-home --shell /bin/bash --uid 1000 researcher
WORKDIR /workspace

# Install pinned dependencies first to use the Docker layer cache
COPY requirements.txt .
RUN pip install -r requirements.txt

# The repository itself is mounted at /workspace (see README), so code edits need no rebuild
USER researcher

# JupyterLab port
EXPOSE 8888

# Token authentication stays ON: the link with the token is printed in the console
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--notebook-dir=/workspace"]
