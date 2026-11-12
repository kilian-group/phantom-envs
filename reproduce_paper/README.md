# Reproducing the paper

This guide rebuilds the datasets, training runs, evaluations, tables and figures of
[PhantomEnvironments: Training LLM Agents in Fictional Worlds](https://arxiv.org/abs/2609.40221). It
assumes the [installation](../README.md#installation) with the `skyrl` extra, SWI-Prolog, and a `.env`
with your W&B settings and run directory. Run every command from the repo root.

[`final_ckpts.yaml`](final_ckpts.yaml) lists the checkpoints behind the paper's results: for each model
and training environment, the training run of each seed and the step we evaluate. Its paths point at our
runs; replace them with your own as you train.

## 1. Data

Every data dir has the same layout: the questions in `tasks/`, the documents the agent searches in
`corpus.jsonl`, and a FAISS index of those documents per retriever: e5-base-v2 (`e5_Flat.index`) for
training and Qwen3-Embedding-4B (`qwen3-4b_Flat.index`) for evaluation.
`scripts/skyrl/prepare_data.py` turns the questions into the parquet files SkyRL reads.

### PhantomEnvs

The three training environments are on [Hugging Face](https://huggingface.co/datasets/kilian-group/phantom-envs).
`generate_paper_data.py` regenerates all six environments with the exact generators that made them and
checks each against the paper's content hash. Hops comes from
[phantom-wiki 1.0.3](https://pypi.org/project/phantom-wiki/1.0.3/); Hops+Constraints and
Hops+Comparisons come from a fork of PhantomWiki at commit
[`839fa26`](https://github.com/swathisaravananan/phantom-wiki/tree/839fa2685d57e6073d604f9eb6ce369a69d34dde).

| Name | Data dir | Universe | Train / val questions | Used for |
|---|---|---|---|---|
| `pw-train` | `data/pw/depth_16_size_1000_seed_1_50k` | 1K people, seed 1 | 56,250 / 6,250 | Hops training |
| `pw1` | `data/pw/depth_16_size_1000_seed_1` | 1K people, seed 1 | 11,250 / 1,250 | Eval on the training universe |
| `pw2` | `data/pw/depth_16_size_1000_seed_2` | 1K people, seed 2 | 11,250 / 1,250 | Eval on an unseen universe |
| `pwOOD` | `data/pw/depth_22_size_10000_seed_2` | 10K people, seed 2 | 16,650 / 1,850 | Eval on an unseen, 10x larger universe |
| `pwcons-train` | `data/pwcons/hops_7_cons_3_size_1000_seed_1_50k` | 1K people, seed 1 | 55,800 / 6,200 | Hops+Constraints training |
| `pwcomp-train` | `data/pwcomp/hops_7_cons_1_size_1000_seed_1_50k_balanced` | 1K people, seed 1 | 53,100 / 5,900 | Hops+Comparisons training |

`pwOOD` and `pwcons-train` need about 94 GB and 62 GB of RAM to generate.

```bash
# Download the training environments into data/
uv run hf download kilian-group/phantom-envs --repo-type dataset --include "*-train.zip" --local-dir data
for z in data/*-train.zip; do unzip -o $z -d data; done

# Regenerate the eval universes (add pw-train, pwcons-train and pwcomp-train to rebuild those too)
for name in pw1 pw2 pwOOD; do
    uv run python reproduce_paper/scripts/generate_paper_data.py $name
done

# Training environments: SkyRL parquets and the e5 index
for d in data/pw/depth_16_size_1000_seed_1_50k \
         data/pwcons/hops_7_cons_3_size_1000_seed_1_50k \
         data/pwcomp/hops_7_cons_1_size_1000_seed_1_50k_balanced; do
    uv run python scripts/skyrl/prepare_data.py --data_dir $d
    uv run --with faiss-cpu==1.12.0 python -m phantom_envs.retriever.build_index --data_dir $d
done

# Eval universes: SkyRL parquets and the qwen3-4b index
for d in data/pw/depth_16_size_1000_seed_1 \
         data/pw/depth_16_size_1000_seed_2 \
         data/pw/depth_22_size_10000_seed_2; do
    uv run python scripts/skyrl/prepare_data.py --data_dir $d
    uv run --with faiss-cpu==1.12.0 python -m phantom_envs.retriever.build_index --data_dir $d --encoder qwen3-4b
done
```

### Search-R1 (NQ+HotpotQA)

The real-world training baseline. Its validation split also holds the PopQA questions of the
[knowledge-preservation evals](#knowledge-preservation).

```bash
# FAISS index (e5_Flat.index, ~64 GB) and the Wikipedia-2018 corpus (corpus.jsonl, ~14 GB)
bash scripts/benchmarks/download_searchr1_index.sh data/searchr1
uv run python scripts/benchmarks/searchr1.py --data_dir data/searchr1
uv run python scripts/skyrl/prepare_data.py --data_dir data/searchr1
```

### Multi-hop benchmarks

Seven real-world benchmarks: HotpotQA (distractor setting), 2WikiMultiHopQA, MuSiQue-Ans, CofCA,
SynthWorlds-RM and SynthWorlds-SM with 500 questions each, and FRAMES with 824. Each benchmark's corpus
is the set of paragraphs that come with its questions (gold and distractors for the first four, gold
only for the SynthWorlds sets). FRAMES uses each question's gold Wikipedia articles as of 2023-06-01,
split into sections.

| Dataset | Benchmark | Source |
|---|---|---|
| `hp500` | HotpotQA | [hp500.zip](https://huggingface.co/datasets/kilian-group/phantom-envs/blob/main/hp500.zip) |
| `2wiki500` | 2WikiMultiHopQA | [2wiki500.zip](https://huggingface.co/datasets/kilian-group/phantom-envs/blob/main/2wiki500.zip) |
| `msq500` | MuSiQue-Ans | [msq500.zip](https://huggingface.co/datasets/kilian-group/phantom-envs/blob/main/msq500.zip) |
| `cofca500` | CofCA | [cofca500.zip](https://huggingface.co/datasets/kilian-group/phantom-envs/blob/main/cofca500.zip) |
| `synthrm500` | SynthWorlds-RM | [synthrm500.zip](https://huggingface.co/datasets/kilian-group/phantom-envs/blob/main/synthrm500.zip) |
| `synthsm500` | SynthWorlds-SM | [synthsm500.zip](https://huggingface.co/datasets/kilian-group/phantom-envs/blob/main/synthsm500.zip) |
| `frames` | FRAMES | [google/frames-benchmark](https://huggingface.co/datasets/google/frames-benchmark), downloaded by `multihop.py` |

```bash
# The question files, unzipped into data/<dataset>/
uv run hf download kilian-group/phantom-envs --repo-type dataset --include "*500.zip" --local-dir data
for z in data/*500.zip; do unzip -o $z -d data; done

for d in hp500 2wiki500 msq500 cofca500 synthrm500 synthsm500 frames; do
    uv run python scripts/benchmarks/multihop.py --dataset $d --data_dir data/$d
    uv run python scripts/skyrl/prepare_data.py --data_dir data/$d
    uv run --with faiss-cpu==1.12.0 python -m phantom_envs.retriever.build_index --data_dir data/$d --encoder qwen3-4b
    uv run --with faiss-cpu==1.12.0 python -m phantom_envs.retriever.build_index --data_dir data/$d
done
```

### Pooled corpus

The corpora of six benchmarks (all but CofCA) in one index, so each question is searched against up to
44x more documents than its own benchmark has. Needs the benchmark dirs above with their qwen3-4b
indexes.

```bash
uv run --with faiss-cpu==1.12.0 python reproduce_paper/scripts/pooled.py --out_dir data/pooled6
uv run python scripts/skyrl/prepare_data.py --data_dir data/pooled6
```

## 2. Training

We train each model with GRPO in each PhantomEnvs environment and on Search-R1; on 2 B200 GPUs a run
takes 1-2 days. The launcher starts the retrieval server, then trains on every GPU of the node. Configs
stack on `scripts/skyrl/configs/base.yaml`, later files winning: a model config from
`scripts/skyrl/configs/models/`, and for Search-R1 `scripts/skyrl/configs/envs/searchr1.yaml`.

```bash
# Hops
bash scripts/skyrl/launch_skyrl.sh train e5 data/pw/depth_16_size_1000_seed_1_50k \
    --config scripts/skyrl/configs/models/Qwen2.5-3B-Instruct.yaml \
    trainer.seed=1

# Hops+Constraints
bash scripts/skyrl/launch_skyrl.sh train e5 data/pwcons/hops_7_cons_3_size_1000_seed_1_50k \
    --config scripts/skyrl/configs/models/Qwen2.5-3B-Instruct.yaml \
    trainer.seed=1

# Hops+Comparisons
bash scripts/skyrl/launch_skyrl.sh train e5 data/pwcomp/hops_7_cons_1_size_1000_seed_1_50k_balanced \
    --config scripts/skyrl/configs/models/Qwen2.5-3B-Instruct.yaml \
    trainer.seed=1
```

Search-R1's 21M-passage index is too slow to search on CPU, so its retriever runs on GPU with faiss-gpu
from conda. Create the conda env once and set `FAISS_GPU_ENV` in `.env` to its path:

```bash
conda create -n phantom-envs-faiss-gpu python=3.12 cuda-version=12.8 \
    "faiss-gpu=1.12.0=hb7818e3_0" "faiss=1.12.0=py312cuda128h4500ccf_0_cuda"

bash scripts/skyrl/launch_skyrl.sh train e5 data/searchr1 \
    --config scripts/skyrl/configs/models/Qwen2.5-3B-Instruct.yaml \
    --config scripts/skyrl/configs/envs/searchr1.yaml \
    retriever.faiss_gpu=true \
    trainer.seed=1
```

Runs go to `$RUN_BASE_DIR/runs/<env>/<model>/grpo/<user>/<date>__r=<reward>__max_t=10__s=<seed>`. Every
10% of training, a run exports the policy in Hugging Face format to `exports/global_step_<N>/policy`,
which the evaluations load.

## 3. Evaluation

Evaluation uses a stronger setup than training: the Qwen3-Embedding-4B retriever, up to 20 search turns
and a 32K-token context, on one GPU (`scripts/skyrl/configs/eval.yaml`, which the launcher stacks last).

### Multi-hop benchmarks

A base model and one of its trained checkpoints on the seven benchmarks, then scoring:

```bash
BENCHMARKS="hp500 2wiki500 msq500 cofca500 synthrm500 synthsm500 frames"
CKPT=$RUN_BASE_DIR/runs/pw/Qwen/Qwen2.5-3B-Instruct/grpo/<user>/<run>/exports/global_step_220/policy
# or our released seed-1 checkpoint: CKPT=kilian-group/PhantomEnvs-Qwen2.5-3B-Instruct

for d in $BENCHMARKS; do
    # Base model
    bash scripts/skyrl/launch_skyrl.sh eval qwen3-4b data/$d \
        trainer.policy.model.path=Qwen/Qwen2.5-3B-Instruct \
        generator.max_input_length=32268 \
        --out_dir=out__eval=${d}__ret=qwen3-4b_Flat

    # Trained checkpoint
    bash scripts/skyrl/launch_skyrl.sh eval qwen3-4b data/$d \
        trainer.policy.model.path=$CKPT \
        --out_dir=out__eval=${d}__ret=qwen3-4b_Flat

    uv run python scripts/score_trajectories.py \
        --eval_dir out__eval=${d}__ret=qwen3-4b_Flat \
        --data_source $d
done
```

Every model evaluated on a benchmark shares its eval dir (`<out_dir>/model=<model>/dumped_evals/`).
`score_trajectories.py` writes EM and F1 for each model to `<out_dir>/scores/<dataset>.csv`; these
are the reported numbers. For Llama and Phi, add their model config for their sampling settings. Base
models take `generator.max_input_length=32268` and checkpoints the default 32768, as in the paper.

Run these evals for every base model and every checkpoint in `final_ckpts.yaml`.

### PhantomWiki universes

The PhantomWiki generalization figure reads these trajectories directly. Its training-curve panels use
the evals that training runs on the fly (`<run>/exports/dumped_evals/`).

```bash
bash scripts/skyrl/launch_skyrl.sh eval qwen3-4b data/pw/depth_16_size_1000_seed_1 \
    trainer.policy.model.path=$CKPT \
    --out_dir=out__eval=pw1__ret=qwen3-4b_Flat

bash scripts/skyrl/launch_skyrl.sh eval qwen3-4b data/pw/depth_16_size_1000_seed_2 \
    trainer.policy.model.path=$CKPT \
    --out_dir=out__eval=pw2__ret=qwen3-4b_Flat

bash scripts/skyrl/launch_skyrl.sh eval qwen3-4b data/pw/depth_22_size_10000_seed_2 \
    trainer.policy.model.path=$CKPT \
    --out_dir=out__eval=pwOOD__ret=qwen3-4b_Flat
```

### Pooled corpus

The pooled eval dir holds questions from six benchmarks, so it is scored once per benchmark:

```bash
bash scripts/skyrl/launch_skyrl.sh eval qwen3-4b data/pooled6 \
    trainer.policy.model.path=$CKPT \
    --out_dir=out__eval=pooled6__ret=qwen3-4b_Flat

for d in hp500 2wiki500 msq500 synthrm500 synthsm500 frames; do
    uv run python scripts/score_trajectories.py \
        --eval_dir out__eval=pooled6__ret=qwen3-4b_Flat \
        --data_source $d
done
```

### Training setup

The retriever tables also evaluate with the training setup: e5-base-v2, 10 turns and an 8192-token
context.

```bash
for d in hp500 2wiki500 msq500 synthrm500 synthsm500 frames; do
    bash scripts/skyrl/launch_skyrl.sh eval e5 data/$d \
        trainer.policy.model.path=$CKPT \
        generator.max_turns=10 \
        generator.max_input_length=8192 \
        --out_dir=out__eval=${d}__ret=e5_Flat

    uv run python scripts/score_trajectories.py \
        --eval_dir out__eval=${d}__ret=e5_Flat \
        --data_source $d
done
```

### F1 over training

The F1 vs. training steps figure evaluates every export of the seed-1 Hops runs, with 10 turns as in
training, and reads the trajectories directly.

```bash
RUN=$RUN_BASE_DIR/runs/pw/Qwen/Qwen2.5-3B-Instruct/grpo/<user>/<run>
for ckpt in $RUN/exports/global_step_*/policy; do
    for d in hp500 2wiki500 msq500 cofca500 synthrm500 synthsm500 frames; do
        bash scripts/skyrl/launch_skyrl.sh eval qwen3-4b data/$d \
            trainer.policy.model.path=$ckpt \
            generator.max_turns=10 \
            --out_dir=out__eval=${d}__all_ckpts
    done
done
```

### Knowledge preservation

Closed-book PopQA, MMLU (with lm-eval) and Wikipedia-2018 perplexity, for the base models and their
Hops and Search-R1 checkpoints, each on one GPU, written to `out__nlu/<eval>/`. PopQA reads its questions
from `data/searchr1/validation.parquet` and its few-shot examples from the PopQA authors' `popQA.tsv`,
which goes in `data/popqa/`.

```bash
# Once: a fixed sample of 1M Wikipedia-2018 passages for perplexity
uv run python reproduce_paper/scripts/sample_wiki_corpus.py \
    --n 1000000 \
    --out data/searchr1/wiki-18.sample1M.jsonl

uv run python reproduce_paper/scripts/eval_popqa.py --model_path $CKPT
uv run python reproduce_paper/scripts/eval_mmlu.py --model_path $CKPT --log_samples
uv run python reproduce_paper/scripts/eval_wiki_ppl.py \
    --model_path $CKPT \
    --corpus_path data/searchr1/wiki-18.sample1M.jsonl
```

## 4. Tables and figures

`save_transfer_data.py` collects the scores of every checkpoint in `final_ckpts.yaml` into JSON files.
The table and figure scripts read those, or the trajectories directly, and write to
`reproduce_paper/figures/`. On our eval outputs they reproduce the paper's tables exactly.

```bash
uv run python reproduce_paper/scripts/save_transfer_data.py

uv run python reproduce_paper/scripts/save_transfer_data.py \
    --retriever e5_Flat \
    --benchmarks hp500,2wiki500,msq500,synthrm500,synthsm500,frames \
    --out_name e5_data.json

uv run python reproduce_paper/scripts/save_transfer_data.py \
    --eval_dir_template "out__eval=pooled6__ret={retriever}" \
    --benchmarks hp500,2wiki500,msq500,synthrm500,synthsm500,frames \
    --out_name pooled_data.json

# CofCA's question categories, including those of the questions the dataset leaves unlabeled
uv run python reproduce_paper/scripts/build_cofca_categories.py
```

| Paper artifact | Command (`uv run python reproduce_paper/scripts/...`) | Reads |
|---|---|---|
| Main results table (`f1_table`) | `create_result_tables.py` | `transfer_data.json` |
| Real vs. synthetic training (`real_vs_synthetic_summary__<model>`) | `plot_real_vs_synthetic_summary.py` | `transfer_data.json` |
| PW generalization (`pw_generalization_vs_difficulty_qwen7b_llama`) | `create_pw_generalization_vs_difficulty.py` | `out__eval=pw{1,2,OOD}__*`, training-time evals |
| PW generalization, appendix (`..._qwen3b_phi`) | `create_pw_generalization_vs_difficulty.py --models Qwen/Qwen2.5-3B-Instruct,microsoft/Phi-4-mini-instruct --fig_suffix _qwen3b_phi` | same |
| F1 vs. training steps (`f1_vs_training_steps`) | `plot_f1_vs_training_steps.py` | `out__eval=*__all_ckpts` |
| Environment variants (`f1_variants_summary`) | `plot_variants_summary.py` | `transfer_data.json` |
| Per-category ΔF1 (`f1_category_delta_pw_variants__{cofca,2wiki}`) | `create_bar_plots_category_delta.py` | trajectories, CofCA categories |
| Pooled corpus (`pooled_table_<model>`) | `create_pooled_tables.py` | `transfer_data.json`, `pooled_data.json` |
| Eval setup (`retriever_table_<model>`) | `create_retriever_tables.py` | `transfer_data.json`, `e5_data.json` |
| Knowledge preservation (`nlu_table`) | `create_nlu_table.py` | `out__nlu/` |

The table scripts also save every number they print as JSON and, if `pdflatex` is installed, a PDF
preview of each table.

## Citation

```bibtex
@article{kabra2026phantomenvironments,
  title={PhantomEnvironments: Training LLM Agents in Fictional Worlds},
  author={Kabra, Anmol and Selvam, Swathi Saravana and Gong, Albert and Wan, Chao and Belardi, Christian and Go, Dongyoung and Luo, Katie Z and Weinberger, Kilian Q},
  journal={arXiv preprint arXiv:2609.40221},
  year={2026}
}
```
