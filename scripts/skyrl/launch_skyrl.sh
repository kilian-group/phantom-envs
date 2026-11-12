#!/usr/bin/env bash
# Run one SkyRL training or eval job on this node, with a retrieval server over <data_dir>.
#
# Usage:
#   bash scripts/skyrl/launch_skyrl.sh <train|eval> <e5|qwen3-4b> <data_dir> [--config <yaml>]... [overrides]...
#
#   train         trains on <data_dir>/{train,validation}.parquet
#   eval          evaluates on <data_dir>/validation.parquet, with configs/eval.yaml last; needs --out_dir=<dir>
#   e5|qwen3-4b   retriever encoder: e5 for training, qwen3-4b for paper evals
#   retriever.*   retriever settings, e.g. retriever.faiss_gpu=true for Search-R1
#   overrides     SkyRL settings in dot notation, and --out_dir / --run_dir / --name
#
# GPUs: CUDA_VISIBLE_DEVICES if set, else the whole node for training and one GPU for an eval.
# Settings come from .env (see .env.example).
#
# Examples:
#   bash scripts/skyrl/launch_skyrl.sh train e5 data/pw/depth_16_size_1000_seed_1_50k trainer.seed=2
#   CUDA_VISIBLE_DEVICES=3 bash scripts/skyrl/launch_skyrl.sh eval qwen3-4b data/hp500 --out_dir=out__eval=hp500 \
#       trainer.policy.model.path=<run>/exports/global_step_160/policy
set -euo pipefail

usage() {
    sed -n '4,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"
launch_setup "$@"

CONFIGS=scripts/skyrl/configs
case $MODE in
    train)
        DATA_ARGS=("data.train_data=[$DATA_DIR/train.parquet]" "data.val_data=[$DATA_DIR/validation.parquet]")
        LAST_CONFIGS=()
        ;;
    eval)
        DATA_ARGS=("data.val_data=[$DATA_DIR/validation.parquet]")
        LAST_CONFIGS=(--config "$CONFIGS/eval.yaml")
        ;;
esac
# Goes before the user's overrides, so an explicit GPU count wins
GPU_ARGS=()
if [ -n "$TRAINER_GPUS" ]; then
    GPU_ARGS=("trainer.placement.policy_num_gpus_per_node=$TRAINER_GPUS")
fi

start_retriever

# Ray's head port stays below the kernel's ephemeral range (32768+)
start_ray 20000 32767 uv run --no-sync ray
uv run --no-sync python "scripts/skyrl/$MODE.py" \
    --config "$CONFIGS/base.yaml" \
    "${DATA_ARGS[@]}" \
    "environment.skyrl_gym.phantom_search.search_url=$SEARCH_URL" \
    "${GPU_ARGS[@]}" \
    "${TRAINER_ARGS[@]}" \
    "${LAST_CONFIGS[@]}"
