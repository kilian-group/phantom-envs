# Image of the Harbor shell tasks (scripts/harbor/export_tasks.py): bash with GNU tools and ripgrep, and Python
# with numpy for the verifier. curl lets Harbor's installed agents (qwen-coder, pi, ...) set themselves up. The
# corpus is not in the image: harbor run mounts it read-only at /corpus.
#
#   docker build -t phantom-envs-shell-task -f scripts/harbor/shell-task.Dockerfile scripts/harbor
FROM python:3.12-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates ripgrep \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir numpy
WORKDIR /app
