#!/usr/bin/env bash
# Train Qwen3.5-9B in the pi harness on PhantomWiki (Hops), with its articles rewritten by an LLM: the
# articles are not on disk, so the agent finds them with a search tool served over MCP, keeps notes with
# bash in a sandbox, and writes its answer to a file. Then evaluate the base and trained models as the
# same agent on six real-world multi-hop QA benchmarks. Runs on one 8xH100 node: training to step 100
# takes ~13 h. Our run raised the average F1 from 56% to 66%.
#
# Training searches with e5, evals with Qwen3-Embedding-4B; the agent sees the same tool either way.
# Training stops at step 100. The warmup is pinned to our run's, which was planned for a full epoch.
# Rewriting the articles calls an LLM on OpenRouter: set OPENROUTER_API_KEY in .env.
#
# Usage, from the repo root after the installation in README.md, including the sandbox image (set CUDA_HOME
# in .env: training compiles Qwen3.5's kernels with nvcc):
#   bash examples/train_pi_search_qwen35-9b.sh [<override>]...    # overrides go to training, e.g. trainer.seed=2
#   CKPT=<run>/exports/global_step_<N>/policy bash examples/train_pi_search_qwen35-9b.sh   # evals only
#
# Data dirs and indexes that are already built are reused. Evals go to out__eval=<benchmark>__pi_search/.
set -euo pipefail

MODEL=Qwen/Qwen3.5-9B
CONFIG=scripts/skyrl/configs/models/Qwen3.5-9B.yaml
ENV_DIR=data/envs/pw-rewrite
BENCHMARKS="hp500 2wiki500 msq500 synthrm500 synthsm500 frames"
GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi
if [ -z "${SANDBOX_ROOTFS:-}" ]; then
    echo "Build the sandbox image (see Installation in README.md) and set SANDBOX_ROOTFS in .env" >&2
    exit 1
fi
# Every rollout in a batch runs its own pi at once, and each holds open files and sockets
ulimit -n "$(ulimit -Hn)"

# 1. Data: the pw-rewrite environment with its e5 index, and the benchmarks with their Qwen3-Embedding-4B
# indexes, all as terminal parquets
[ -f $ENV_DIR/corpus.jsonl ] || uv run phantom_envs generate --preset pw --output_dir $ENV_DIR --overwrite \
    rewrite_docs_with_llm=openai/gpt-6-luna
[ -f $ENV_DIR/terminal/train.parquet ] || uv run python scripts/skyrl/prepare_data.py --data_dir $ENV_DIR --env terminal
[ -f $ENV_DIR/e5_Flat.index ] || uv run --with faiss-cpu==1.12.0 python -m phantom_envs.retriever.build_index \
    --data_dir $ENV_DIR --encoder e5
for d in $BENCHMARKS; do
    if [ ! -f data/$d/corpus.jsonl ]; then
        # multihop.py downloads FRAMES itself
        if [ $d != frames ]; then
            uv run hf download kilian-group/phantom-envs $d.zip --repo-type dataset --local-dir data
            unzip -o data/$d.zip -d data
        fi
        uv run python scripts/benchmarks/multihop.py --dataset $d --data_dir data/$d
    fi
    [ -f data/$d/terminal/validation.parquet ] || uv run python scripts/skyrl/prepare_data.py --data_dir data/$d --env terminal
    [ -f data/$d/qwen3-4b_Flat.index ] || uv run --with faiss-cpu==1.12.0 python -m phantom_envs.retriever.build_index \
        --data_dir data/$d --encoder qwen3-4b
done

# 2. Train on every GPU: 100 steps of 128 questions x 8 rollouts, exporting the policy every 20 steps.
# --search runs the agent as pi in bubblewrap, with bash and the search tool. The e5 search server
# shares GPU 0 with the trainer; a smaller micro-batch token budget leaves it room.
if [ -z "${CKPT:-}" ]; then
    CUDA_VISIBLE_DEVICES=$GPUS bash scripts/skyrl/launch_skyrl.sh train terminal $ENV_DIR \
        --search=e5 \
        --config $CONFIG \
        --run_dir examples_search \
        trainer.train_batch_size=128 \
        trainer.policy_mini_batch_size=128 \
        trainer.max_tokens_per_microbatch=12288 \
        trainer.max_training_steps=100 \
        trainer.policy.optimizer_config.num_warmup_steps=38 \
        trainer.eval_interval=50 \
        trainer.hf_save_interval=20 \
        trainer.ckpt_interval=50 \
        "$@"
    RUN=$(ls -td "${RUN_BASE_DIR:-.}/examples_search/pw/$MODEL/grpo/$USER"/*/ | head -1)
    CKPT=${RUN}exports/global_step_100/policy
fi

# 3. Evaluate both models on every GPU (the model config runs 4 vLLM engines with tensor parallelism 2),
# searching with Qwen3-Embedding-4B
for d in $BENCHMARKS; do
    for model in $MODEL "$CKPT"; do
        CUDA_VISIBLE_DEVICES=$GPUS bash scripts/skyrl/launch_skyrl.sh eval terminal data/$d \
            --search=qwen3-4b \
            --config $CONFIG \
            trainer.policy.model.path=$model \
            --out_dir=out__eval=${d}__pi_search
    done
    uv run python scripts/score_trajectories.py --eval_dir out__eval=${d}__pi_search --data_source $d
done

# Average F1 over the benchmarks, per model
uv run python - $BENCHMARKS <<'EOF'
import sys

import pandas as pd

scores = pd.concat(pd.read_csv(f"out__eval={d}__pi_search/scores/{d}.csv") for d in sys.argv[1:])
print((100 * scores.groupby("model")["f1"].mean()).round(1).to_string())
EOF
