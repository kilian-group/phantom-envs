#!/usr/bin/env bash
# Run one SkyRL training or eval job on this node, with a retrieval server over <data_dir>, or with
# the terminal env: a coding agent in a sandbox, with bash over <data_dir>/corpus.jsonl or a search tool.
#
# Usage:
#   bash scripts/skyrl/launch_skyrl.sh <train|eval> <e5|qwen3-4b|terminal> <data_dir> [--search=<r>] [--config <yaml>]... [overrides]...
#
#   train         trains on <data_dir>/{train,validation}.parquet
#   eval          evaluates on <data_dir>/validation.parquet, with configs/eval.yaml last; needs --out_dir=<dir>
#   e5|qwen3-4b   retriever encoder: e5 for training, qwen3-4b for paper evals
#   terminal      the phantom_terminal env: <data_dir>/terminal/*.parquet (prepare_data.py --env terminal), with
#                 configs/envs/shellsim.yaml right after base.yaml. Its sandbox is ShellSim, or bubblewrap
#                 with environment.skyrl_gym.phantom_terminal.sandbox=bwrap (needs SANDBOX_ROOTFS)
#   --search=<r>  with terminal: give the agent a search tool instead of the corpus as files, answered
#                 by retriever <r> (bm25, e5 or qwen3-4b) through an MCP server. The agent is then pi in
#                 bubblewrap (needs SANDBOX_ROOTFS), with bash and the search tool.
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
#   bash scripts/skyrl/launch_skyrl.sh eval terminal data/hp500 --config scripts/skyrl/configs/models/Qwen3.5-9B.yaml \
#       --out_dir=out__eval=hp500__terminal
set -euo pipefail

usage() {
    sed -n '5,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

source "$(dirname "${BASH_SOURCE[0]}")/../launch_common.sh"
launch_setup "$@"
if [ -n "$SEARCH" ] && [ "$ENCODER" != terminal ]; then
    echo "--search gives the terminal agent a search tool: use it with terminal, not $ENCODER" >&2
    exit 1
fi
if [ -n "$SEARCH" ] && [ -z "${SANDBOX_ROOTFS:-}" ]; then
    echo "--search runs pi in bubblewrap: build the sandbox image (see README.md) and set SANDBOX_ROOTFS" >&2
    exit 1
fi

CONFIGS=scripts/skyrl/configs
PARQUET_DIR=$DATA_DIR
ENV_ARGS=()
if [ "$ENCODER" = terminal ]; then
    PARQUET_DIR=$DATA_DIR/terminal
    # Absolute: Ray runs the entrypoint from its own working dir
    ENV_ARGS=(--config "$CONFIGS/envs/shellsim.yaml"
        "environment.skyrl_gym.phantom_terminal.corpus_path=$(cd "$DATA_DIR" && pwd)/corpus.jsonl")
    # The bwrap sandbox's image, used with environment.skyrl_gym.phantom_terminal.sandbox=bwrap
    if [ -n "${SANDBOX_ROOTFS:-}" ]; then
        ENV_ARGS+=("environment.skyrl_gym.phantom_terminal.rootfs=$SANDBOX_ROOTFS")
    fi
fi
case $MODE in
    train)
        DATA_ARGS=("data.train_data=[$PARQUET_DIR/train.parquet]" "data.val_data=[$PARQUET_DIR/validation.parquet]")
        LAST_CONFIGS=()
        ;;
    eval)
        DATA_ARGS=("data.val_data=[$PARQUET_DIR/validation.parquet]")
        LAST_CONFIGS=(--config "$CONFIGS/eval.yaml")
        ;;
esac
# Goes before the user's overrides, so an explicit GPU count wins
GPU_ARGS=()
if [ -n "$TRAINER_GPUS" ]; then
    GPU_ARGS=("trainer.placement.policy_num_gpus_per_node=$TRAINER_GPUS")
fi

if [ "$ENCODER" != terminal ]; then
    start_retriever rest_server "$ENCODER"
    ENV_ARGS=("environment.skyrl_gym.phantom_search.search_url=http://127.0.0.1:$RETRIEVER_PORT/retrieve")
elif [ -n "$SEARCH" ]; then
    # Sandboxes have no network: they reach the MCP server on this socket, which is mounted into them
    MCP_SOCKET=/tmp/phantom-mcp-$RUN_ID.sock
    start_retriever mcp_server "$SEARCH" "socket=$MCP_SOCKET"
    # pi is the harness that speaks MCP. Before the user's overrides, which may pick other tools.
    ENV_ARGS+=("environment.skyrl_gym.phantom_terminal.sandbox=bwrap"
        "environment.skyrl_gym.phantom_terminal.harness=pi"
        "environment.skyrl_gym.phantom_terminal.tools=bash+search"
        "environment.skyrl_gym.phantom_terminal.mcp_socket=$MCP_SOCKET")
fi

# Ray's head port stays below the kernel's ephemeral range (32768+)
start_ray 20000 32767 uv run --no-sync ray
uv run --no-sync python "scripts/skyrl/$MODE.py" \
    --config "$CONFIGS/base.yaml" \
    "${ENV_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${GPU_ARGS[@]}" \
    "${TRAINER_ARGS[@]}" \
    "${LAST_CONFIGS[@]}"
