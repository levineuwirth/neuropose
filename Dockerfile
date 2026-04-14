# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# NeuroPose — CPU Docker image.
#
# Builds a minimal CPU-only image running the NeuroPose job-processing
# daemon (`neuropose watch`). The GPU variant is planned but not
# shipped in v0.1 — it will live in a separate `Dockerfile.gpu` based
# on a CUDA-enabled TensorFlow base image.
#
# Build:
#   docker build -t neuropose:latest .
#
# Run (daemon mode, the default):
#   docker run -d \
#     -v /srv/neuropose/jobs:/data/jobs \
#     -v /srv/neuropose/models:/data/models \
#     --name neuropose \
#     neuropose:latest
#
# Run (single-video mode, overriding the default command):
#   docker run --rm \
#     -v /srv/neuropose/jobs:/data/jobs \
#     -v /srv/neuropose/models:/data/models \
#     -v $PWD/video.mp4:/input.mp4:ro \
#     neuropose:latest process /input.mp4 --output /data/jobs/result.json
#
# Notes:
# - The Python base image (`python:3.11-slim-bookworm`) is deliberately
#   not pinned to a specific patch version in this commit. A sha256
#   digest pin is an easy follow-up once we are ready to commit to a
#   reproducible build chain.
# - TensorFlow will be downloaded by pip during the build (~500 MB). The
#   final image is correspondingly large; optimisation is a future
#   concern.
# - The MeTRAbs model itself is NOT baked into the image. It downloads
#   on first daemon startup into /data/models, which must be mounted
#   from the host to avoid repeating the download on every container
#   start.
# ---------------------------------------------------------------------------

FROM python:3.11-slim-bookworm AS runtime

# System dependencies:
# - ca-certificates: HTTPS for the MeTRAbs model download.
# - ffmpeg:          video I/O backend OpenCV calls into.
# - libgl1, libglib2.0-0: transitive requirements of
#                         opencv-python-headless on slim images.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      ffmpeg \
      libgl1 \
      libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package metadata first so that source-only edits do not bust
# the pip cache layer on rebuilds.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Install the package system-wide. The `[analysis]` extra pulls in
# fastdtw / scipy / scikit-learn / sktime so `neuropose analyze`
# works out of the box inside the container.
RUN pip install --no-cache-dir ".[analysis]"

# Non-root runtime user. Using UID 1000 to line up with most host-side
# /data directories; override with `docker run --user` if needed.
RUN useradd --create-home --uid 1000 neuropose \
 && mkdir -p /data/jobs /data/models \
 && chown -R neuropose:neuropose /data

# Point NeuroPose's Settings at the mounted /data volume. These are
# read by pydantic-settings via the NEUROPOSE_ prefix and override the
# XDG defaults that would otherwise resolve inside the container's
# ephemeral filesystem.
ENV NEUROPOSE_DATA_DIR=/data/jobs \
    NEUROPOSE_MODEL_CACHE_DIR=/data/models \
    PYTHONUNBUFFERED=1

USER neuropose
VOLUME ["/data"]

# The entrypoint is the `neuropose` CLI; the default command is
# `watch` (daemon mode). Override at `docker run` time to invoke
# `process` or `analyze` instead.
ENTRYPOINT ["neuropose"]
CMD ["watch"]
