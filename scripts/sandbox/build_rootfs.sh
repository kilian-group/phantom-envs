#!/usr/bin/env bash
# Build the sandbox image and export its root filesystem for the bwrap sandbox.
#
# Usage:
#   bash scripts/sandbox/build_rootfs.sh <rootfs_dir>
#
# Needs Docker. <rootfs_dir> must not exist yet: running sandboxes mount it, so build a new image
# into a new dir. Then point SANDBOX_ROOTFS in .env at it.
#
# Example:
#   bash scripts/sandbox/build_rootfs.sh /data/shared/phantom-envs/sandbox-rootfs
set -euo pipefail

if [ $# -ne 1 ]; then
    sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 1
fi
ROOTFS=$1
IMAGE=phantom-envs-sandbox
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ -e "$ROOTFS" ]; then
    echo "Error: $ROOTFS exists; export to a new dir"
    exit 1
fi

docker build -t "$IMAGE" "$HERE"

container=$(docker create "$IMAGE")
trap 'docker rm "$container" > /dev/null' EXIT
mkdir -p "$ROOTFS"
# Device nodes can't be made without root, and the sandbox has its own /dev
docker export "$container" | tar -x -C "$ROOTFS" --no-same-owner --exclude='dev/*'

echo "Exported $IMAGE ($(docker image inspect --format '{{.Id}}' "$IMAGE")) to $ROOTFS"
