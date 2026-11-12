# Running on Cornell's unicorn cluster

These scripts run the SkyRL launcher as SLURM jobs on unicorn (partition `aimi`, B200 nodes).
Anywhere else, run `bash scripts/skyrl/launch_skyrl.sh` directly with a `.env` made from `.env.example`.

## Set up a checkout (once)

```bash
bash scripts/clusters/create_env_unicorn.sh
```

It asks for your W&B settings and an optional conda env with faiss-gpu (for Search-R1's retriever),
writes `.env` with run and cache dirs on `/scratch`, and links:

- `data/` to the shared datasets in `/share/aimi/phantom-agent/data`, which already hold every data dir
  of [reproduce_paper/README.md](../../reproduce_paper/README.md) with its indexes;
- `scratch/` to your run dir, and `share/` to the shared runs.

## Submit a job

```bash
# Training: 2 GPUs, 256 GB, 48 h by default
sbatch scripts/clusters/launch_unicorn.sh \
    scripts/skyrl/launch_skyrl.sh train e5 data/pw/depth_16_size_1000_seed_1_50k

# An eval: sbatch flags override the defaults
sbatch --gres=gpu:1 --ntasks=16 --mem=128GB --time=06:00:00 \
    scripts/clusters/launch_unicorn.sh \
    scripts/skyrl/launch_skyrl.sh eval qwen3-4b data/hp500 --out_dir=out__eval=hp500
```

Job logs go to `logs/`. `/scratch` is local to each node, so pin a job to the node that holds its run
dir and caches with `sbatch -w <node>`.
