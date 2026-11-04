# Reproducible environment for: ai-companion-markers
# CPU-only (no CUDA needed). Python 3.11.9 and OS pinned via the base image tag.
FROM python:3.11.9-slim-bookworm

# Deterministic, quiet Python runtime behaviour
ENV PYTHONHASHSEED=42 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=UTC

# Install minimal OS dependencies for compilation and git tooling
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        make \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Non-root unprivileged user (avoids root-owned files in mounted volumes)
RUN useradd --create-home --shell /bin/bash --uid 1000 researcher
WORKDIR /workspace

# Install pinned dependencies first to leverage Docker layer caching
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files and ensure non-root ownership
COPY . /workspace
RUN chown -R researcher:researcher /workspace

# Switch to unprivileged researcher user
USER researcher

# JupyterLab default port
EXPOSE 8888

# Default entrypoint starts JupyterLab without root privileges.
# Can be overridden via CLI, e.g.:
# docker run --rm -v "$(pwd)":/workspace ai-companion-markers bash scripts/run_all.sh --sample
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--notebook-dir=/workspace"]
