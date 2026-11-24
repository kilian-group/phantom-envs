# Examples

Each script generates its training data, trains a model with RL, and evaluates the base and trained
models as the same agent on six real-world multi-hop QA benchmarks: HotpotQA, 2WikiMultiHopQA, MuSiQue,
SynthRM, SynthSM and FRAMES. Run them from the repo root after the [installation](../README.md#installation);
each script's header lists what else it needs.

```bash
# Data, training and evals
bash examples/train_pi_search_qwen35-9b.sh
# Evals only, of a checkpoint you trained
CKPT=<run>/exports/global_step_<N>/policy bash examples/train_pi_search_qwen35-9b.sh
```

## Results

Average F1 over the six benchmarks, in each agent harness:

| Model | F1 | Training |
|---|---|---|
| **pi + bash + search over MCP** | | |
| Qwen3.5-9B | 56 | |
| [Qwen3.5-9B + PhantomEnvs](train_pi_search_qwen35-9b.sh) | 65 | 9 h on 8xH100 |
| **pi + bash** ([visualization](../assets/terminal-agent.gif)) | | |
| Qwen3.5-9B | 46 | |
| [Qwen3.5-9B + PhantomEnvs](train_pi_qwen35-9b.sh) | 62 | 15 h on 8xH100 |
| Qwen4-Max | 69 | |
| Inkling 2 Small | 65 | |
| [Inkling 2 Small + PhantomEnvs](train_inkling_tinker.sh) | 75 | 6 h, ~$145 with Tinker |
| **ShellSim + bash** | | |
| Qwen3.5-9B | 50 | |
| [Qwen3.5-9B + PhantomEnvs](train_shell_qwen35-9b.sh) | 63 | 11 h on 8xH100 |
| Qwen4-Max | 65 | |
| **Search-R1 style** | | |
| Qwen3-30B-A3B | 46 | |
| [Qwen3-30B-A3B + PhantomEnvs](train_pw_qwen3-30b.sh) | 56 | 40 h on 8xH100 |
