#!/usr/bin/env bash
# Train Qwen3-30B-A3B-Instruct-2507 on PhantomWiki (Hops) with SkyRL's Megatron backend, then evaluate
# the base and trained models on six real-world multi-hop QA benchmarks. Runs on one 8xH100 node:
# training takes ~40 h, and in our run raised the average F1 from 46% to 56%.
#
# Usage, from the repo root after the installation in README.md:
#   bash examples/train_pw_qwen3-30b.sh [<override>]...    # overrides go to training, e.g. trainer.seed=2
#   CKPT=<run>/exports/global_step_<N>/policy bash examples/train_pw_qwen3-30b.sh   # evals only
#
# Data dirs that are already built are reused. Evals go to out__eval=<benchmark>/.
set -euo pipefail

MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
ENV_DIR=data/envs/pw
BENCHMARKS="hp500 2wiki500 msq500 synthrm500 synthsm500 frames"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

build_index() {
    uv run --with faiss-cpu==1.12.0 python -m phantom_envs.retriever.build_index "$@"
}

# 1. Data: the pw environment with its e5 index for training, and the benchmarks with their
#    Qwen3-Embedding-4B indexes for evaluation (as in reproduce_paper/README.md)
if [ ! -f $ENV_DIR/e5_Flat.index ]; then
    uv run phantom_envs generate --preset pw --output_dir $ENV_DIR --overwrite
    uv run python scripts/skyrl/prepare_data.py --data_dir $ENV_DIR
    build_index --data_dir $ENV_DIR
fi
for d in $BENCHMARKS; do
    [ -f data/$d/qwen3-4b_Flat.index ] && continue
    # multihop.py downloads FRAMES itself
    if [ $d != frames ]; then
        uv run hf download kilian-group/phantom-envs $d.zip --repo-type dataset --local-dir data
        unzip -o data/$d.zip -d data
    fi
    uv run python scripts/benchmarks/multihop.py --dataset $d --data_dir data/$d
    uv run python scripts/skyrl/prepare_data.py --data_dir data/$d
    build_index --data_dir data/$d --encoder qwen3-4b
done

# 2. Train on every GPU. The run exports the policy every 10% of training; evaluate the last export.
if [ -z "${CKPT:-}" ]; then
    bash scripts/skyrl/launch_skyrl.sh train e5 $ENV_DIR \
        --config scripts/skyrl/configs/models/Qwen3-30B-A3B-Instruct-2507.yaml \
        --run_dir examples \
        "$@"
    RUN=$(ls -td "${RUN_BASE_DIR:-.}/examples/pw/$MODEL/grpo/$USER"/*/ | head -1)
    CKPT=$(ls -d "$RUN"exports/global_step_*/policy | sort -V | tail -1)
fi

# 3. Evaluate both models on every GPU: 4 vLLM engines with tensor parallelism 2
for d in $BENCHMARKS; do
    for model in $MODEL "$CKPT"; do
        bash scripts/skyrl/launch_skyrl.sh eval qwen3-4b data/$d \
            trainer.policy.model.path=$model \
            generator.inference_engine.num_engines=4 \
            generator.inference_engine.tensor_parallel_size=2 \
            --out_dir=out__eval=$d
    done
    uv run python scripts/score_trajectories.py --eval_dir out__eval=$d --data_source $d
done

# Average F1 over the benchmarks, per model
uv run python - $BENCHMARKS <<'EOF'
import sys

import pandas as pd

scores = pd.concat(pd.read_csv(f"out__eval={d}/scores/{d}.csv") for d in sys.argv[1:])
print((100 * scores.groupby("model")["f1"].mean()).round(1).to_string())
EOF
