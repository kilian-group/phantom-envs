#!/usr/bin/env bash
# Train Qwen3.5-9B as a coding agent on PhantomWiki (Hops): in a ShellSim sandbox, it searches the corpus as
# files with bash and writes its answer to a file. Then evaluate the base and trained models as the same
# agent on six real-world multi-hop QA benchmarks. Runs on one 8xH100 node: training takes ~11 h, and in our
# run raised the average F1 from 50% to 63%.
#
# Training stops at step 120: PhantomWiki validation F1 saturates near 98% by then, and later steps
# transfer worse. The warmup is pinned to our run's (10% of the full 386 steps).
#
# Usage, from the repo root after the installation in README.md (set CUDA_HOME in .env: training compiles
# Qwen3.5's kernels with nvcc):
#   bash examples/train_shell_qwen35-9b.sh [<override>]...    # overrides go to training, e.g. trainer.seed=2
#   CKPT=<run>/exports/global_step_<N>/policy bash examples/train_shell_qwen35-9b.sh   # evals only
#
# Data dirs that are already built are reused. Evals go to out__eval=<benchmark>__terminal/.
set -euo pipefail

MODEL=Qwen/Qwen3.5-9B
CONFIG=scripts/skyrl/configs/models/Qwen3.5-9B.yaml
ENV_DIR=data/envs/pw
BENCHMARKS="hp500 2wiki500 msq500 synthrm500 synthsm500 frames"
GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# 1. Data: the pw environment and the benchmarks as terminal parquets (the terminal style needs no index)
# 3 friends per person, as in our run (the default is 0.5)
[ -f $ENV_DIR/corpus.jsonl ] || uv run phantom_envs generate --preset pw --output_dir $ENV_DIR --overwrite \
    --friendship_degree 3
[ -f $ENV_DIR/terminal/train.parquet ] || uv run python scripts/skyrl/prepare_data.py --data_dir $ENV_DIR --env terminal
for d in $BENCHMARKS; do
    [ -f data/$d/terminal/validation.parquet ] && continue
    if [ ! -f data/$d/corpus.jsonl ]; then
        # multihop.py downloads FRAMES itself
        if [ $d != frames ]; then
            uv run hf download kilian-group/phantom-envs $d.zip --repo-type dataset --local-dir data
            unzip -o data/$d.zip -d data
        fi
        uv run python scripts/benchmarks/multihop.py --dataset $d --data_dir data/$d
    fi
    uv run python scripts/skyrl/prepare_data.py --data_dir data/$d --env terminal
done

# 2. Train on every GPU: 120 steps of 128 questions x 8 rollouts, exporting the policy every 40 steps
if [ -z "${CKPT:-}" ]; then
    CUDA_VISIBLE_DEVICES=$GPUS bash scripts/skyrl/launch_skyrl.sh train terminal $ENV_DIR \
        --config $CONFIG \
        --run_dir examples \
        trainer.train_batch_size=128 \
        trainer.policy_mini_batch_size=128 \
        trainer.max_training_steps=120 \
        trainer.policy.optimizer_config.num_warmup_steps=38 \
        trainer.eval_interval=40 \
        trainer.hf_save_interval=40 \
        trainer.ckpt_interval=40 \
        "$@"
    RUN=$(ls -td "${RUN_BASE_DIR:-.}/examples/pw/$MODEL/grpo/$USER"/*/ | head -1)
    CKPT=${RUN}exports/global_step_120/policy
fi

# 3. Evaluate both models on every GPU (the model config runs 4 vLLM engines with tensor parallelism 2)
for d in $BENCHMARKS; do
    for model in $MODEL "$CKPT"; do
        CUDA_VISIBLE_DEVICES=$GPUS bash scripts/skyrl/launch_skyrl.sh eval terminal data/$d \
            --config $CONFIG \
            trainer.policy.model.path=$model \
            --out_dir=out__eval=${d}__terminal
    done
    uv run python scripts/score_trajectories.py --eval_dir out__eval=${d}__terminal --data_source $d
done

# Average F1 over the benchmarks, per model
uv run python - $BENCHMARKS <<'EOF'
import sys

import pandas as pd

scores = pd.concat(pd.read_csv(f"out__eval={d}__terminal/scores/{d}.csv") for d in sys.argv[1:])
print((100 * scores.groupby("model")["f1"].mean()).round(1).to_string())
EOF
