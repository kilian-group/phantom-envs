# The image of the terminal sandbox: bash with GNU tools, ripgrep and Python, plus Node.js and the pi
# coding agent. The corpus is not in the image; it is mounted read-only at /corpus.
#
# Two things run it:
#   - the bwrap sandbox, from the root filesystem that build_rootfs.sh exports;
#   - Harbor's Docker tasks (scripts/harbor/export_tasks.py), whose verifier needs Python and numpy,
#     and whose installed agents need curl.
#
# Base images and pi are pinned; Debian packages follow bookworm's stable updates.
FROM node:22-bookworm-slim@sha256:e9f565d32859f0cbfb52125eafe8be39390f24dac302dc3cdf566e8ba3ab21ee AS node

FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gawk jq ripgrep \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir numpy==2.2.6

COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && npm install -g @earendil-works/pi-coding-agent@1.5.2 \
    && npm cache clean --force

WORKDIR /app
