#!/usr/bin/env bash
# Fine-tune Inkling 2 Small with Tinker in the pi harness on PhantomWiki (Hops), with its articles
# rewritten by an LLM: with bash in a sandbox, it searches the corpus as files and writes its answer to a
# file. Then evaluate the base and fine-tuned models as the same agent on six real-world multi-hop QA
# benchmarks. Tinker hosts the model and trains a LoRA adapter; the sandboxes and pi run on this machine,
# which needs no GPU. Our run took ~6 h and cost ~$145 on Tinker, and raised the average F1 from 65% to 75%.
#
# Set TINKER_API_KEY and SANDBOX_ROOTFS in .env (see Installation in README.md for the sandbox image).
# Rewriting the articles calls an LLM on OpenRouter: set OPENROUTER_API_KEY too.
#
# Usage, from the repo root after `uv sync --extra skyrl --extra tinker`:
#   bash examples/train_inkling_tinker.sh [<override>]...   # overrides go to training, e.g. max_steps=20
#   CKPT=tinker://<run>/sampler_weights/final bash examples/train_inkling_tinker.sh   # evals only
#
# Data dirs that are already built are reused. Evals go to out__eval=<benchmark>__pi/.
set -euo pipefail

CONFIG=scripts/tinker/configs/models/Inkling-2-Small.yaml
ENV_DIR=data/envs/pw-rewrite
BENCHMARKS="hp500 2wiki500 msq500 synthrm500 synthsm500 frames"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi
if [ -z "${TINKER_API_KEY:-}" ] || [ -z "${SANDBOX_ROOTFS:-}" ]; then
    echo "Set TINKER_API_KEY and SANDBOX_ROOTFS in .env (see Installation in README.md)" >&2
    exit 1
fi
# Every episode of a training step runs its own pi at once, and each holds open files and sockets
ulimit -n "$(ulimit -Hn)"

# 1. Data: the pw-rewrite environment and the benchmarks (Tinker reads their tasks; no parquets)
[ -f $ENV_DIR/corpus.jsonl ] || uv run phantom_envs generate --preset pw --output_dir $ENV_DIR --overwrite \
    rewrite_docs_with_llm=openai/gpt-6-luna
for d in $BENCHMARKS; do
    [ -f data/$d/corpus.jsonl ] && continue
    # multihop.py downloads FRAMES itself
    if [ $d != frames ]; then
        uv run hf download kilian-group/phantom-envs $d.zip --repo-type dataset --local-dir data
        unzip -o data/$d.zip -d data
    fi
    uv run python scripts/benchmarks/multihop.py --dataset $d --data_dir data/$d
done

# 2. Train: 50 steps of 32 questions x 8 episodes, evaluating on PhantomWiki every 10 steps
if [ -z "${CKPT:-}" ]; then
    RUN=${RUN_BASE_DIR:-.}/examples/tinker/inkling-pw-rewrite-$(date +%m-%d-%H%M)
    uv run python scripts/tinker/train.py \
        --config $CONFIG \
        data_dir=$ENV_DIR \
        log_path=$RUN \
        behavior_if_log_dir_exists=raise \
        "$@"
    CKPT=$(uv run python -c "
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
print(next(row['sampler_path'] for row in rows if row['name'] == 'final'))" $RUN/checkpoints.jsonl)
fi

# 3. Evaluate both models on every benchmark
for d in $BENCHMARKS; do
    uv run python scripts/tinker/eval.py --config $CONFIG data_dir=data/$d out_dir=out__eval=${d}__pi \
        name=Inkling-2-Small
    uv run python scripts/tinker/eval.py --config $CONFIG data_dir=data/$d out_dir=out__eval=${d}__pi \
        checkpoint=$CKPT name=Inkling-2-Small__tinker
done

# Average F1 over the benchmarks, per model
uv run python - $BENCHMARKS <<'EOF'
import json
import sys

for model in ("Inkling-2-Small", "Inkling-2-Small__tinker"):
    f1 = [json.load(open(f"out__eval={d}__pi/model={model}/aggregated_results.jsonl"))["f1"] for d in sys.argv[1:]]
    print(f"{model}: {100 * sum(f1) / len(f1):.1f}")
EOF
