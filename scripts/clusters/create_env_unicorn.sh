#!/usr/bin/env bash
# Set up this checkout for the unicorn cluster: write .env and link the shared data and scratch dirs.
#
#   .env      RUN_BASE_DIR and HF_HOME on /scratch, W&B settings and FAISS_GPU_ENV (prompted; it can
#             create a conda env with faiss-gpu for Search-R1's retriever)
#   data    -> /share/aimi/phantom-agent/data    prepared datasets and indexes
#   share   -> /share/aimi/phantom-agent         shared runs and models
#   scratch -> $RUN_BASE_DIR                     your checkpoints and HF exports
#   logs/                                        SLURM job logs of scripts/clusters/launch_unicorn.sh
#
# Usage:
#   bash scripts/clusters/create_env_unicorn.sh    (refuses to overwrite an existing .env)
set -euo pipefail

SHARE_DIR=/share/aimi/phantom-agent
RUN_BASE_DIR=/scratch/$USER/phantom-agent
HF_HOME=/scratch/$USER/huggingface

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
if [ -e .env ]; then
    echo "Error: $REPO/.env already exists; move it away to regenerate it."
    exit 1
fi

read -rp "WANDB_ENTITY (W&B organization): " WANDB_ENTITY
read -rp "WANDB_PROJECT [phantom-envs]: " WANDB_PROJECT
WANDB_PROJECT=${WANDB_PROJECT:-phantom-envs}
read -rsp "WANDB_API_KEY (input is hidden): " WANDB_API_KEY
echo

# Search-R1's 21M-doc index needs faiss-gpu, which only conda ships for B200 (CUDA 12.8 builds).
FAISS_GPU_ENV_NAME=phantom-envs-faiss-gpu
read -rp "FAISS_GPU_ENV (conda env with faiss-gpu): a path, 'new' to create $FAISS_GPU_ENV_NAME, or Enter to skip: " FAISS_GPU_ENV
if [ "$FAISS_GPU_ENV" = "new" ]; then
    conda create -y -n "$FAISS_GPU_ENV_NAME" python=3.12 cuda-version=12.8 \
        "faiss-gpu=1.12.0=hb7818e3_0" "faiss=1.12.0=py312cuda128h4500ccf_0_cuda"
    FAISS_GPU_ENV=$(conda info --base)/envs/$FAISS_GPU_ENV_NAME
fi
if [ -n "$FAISS_GPU_ENV" ] && [ ! -d "$FAISS_GPU_ENV/lib/python3.12/site-packages/faiss" ]; then
    echo "Warning: no faiss in $FAISS_GPU_ENV/lib/python3.12/site-packages"
fi

umask 077  # .env holds the W&B key
cat > .env << EOF
# Written by scripts/clusters/create_env_unicorn.sh
RUN_BASE_DIR=$RUN_BASE_DIR
HF_HOME=$HF_HOME
WANDB_ENTITY=$WANDB_ENTITY
WANDB_PROJECT=$WANDB_PROJECT
WANDB_API_KEY=$WANDB_API_KEY
FAISS_GPU_ENV=$FAISS_GPU_ENV
EOF
umask 022
echo "Wrote $REPO/.env"

# Link <name> -> <target>, leaving an existing file or link alone.
link() {
    local target=$1 name=$2
    if [ -L "$name" ] && [ "$(readlink "$name")" = "$target" ]; then
        echo "ok:      $name -> $target"
    elif [ -e "$name" ] || [ -L "$name" ]; then
        echo "skipped: $name already exists (expected a link to $target)"
    else
        ln -s "$target" "$name"
        echo "linked:  $name -> $target"
    fi
}
mkdir -p "$RUN_BASE_DIR" "$HF_HOME" logs
link "$SHARE_DIR/data" data
link "$SHARE_DIR" share
link "$RUN_BASE_DIR" scratch
