#!/usr/bin/env -S uv run python
"""Write a bf16 copy of an fp32 HuggingFace checkpoint dir, at half the disk space.

SkyRL's HF exports (``<run>/exports/global_step_<N>/policy``) hold fp32 weights, which vLLM serves in
bf16 anyway.
fp32 tensors are cast to bf16 and config.json's dtype is set to bfloat16; every other file
(tokenizer, generation config, ...) is copied as is.

Usage:
    uv run python scripts/convert_hf_fp32_to_bf16.py \
        --input_dir <run>/exports/global_step_160/policy \
        --output_dir <run>/exports_bf16/global_step_160/policy
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# config.json keys that hold the weights' dtype (newer transformers write "dtype")
DTYPE_KEYS = ("torch_dtype", "dtype")


def convert_shard(src: Path, dst: Path) -> int:
    """Cast a safetensors file's fp32 tensors to bf16.

    Args:
        src: The fp32 file.
        dst: Where to write the bf16 file.

    Returns:
        The bytes of tensor data written.
    """
    tensors = load_file(src)
    for name, tensor in tensors.items():
        if tensor.dtype == torch.float32:
            tensors[name] = tensor.to(torch.bfloat16)
    save_file(tensors, dst, metadata={"format": "pt"})
    return sum(t.numel() * t.element_size() for t in tensors.values())


def write_config(src: Path, dst: Path) -> None:
    """Copy config.json with its dtype set to bfloat16 (adding torch_dtype when it has none)."""
    config = json.loads(src.read_text())
    keys = [k for k in DTYPE_KEYS if k in config] or ["torch_dtype"]
    for key in keys:
        config[key] = "bfloat16"
    dst.write_text(json.dumps(config, indent=2) + "\n")


def main(args: argparse.Namespace) -> None:
    """Convert every safetensors file of the input dir and copy the rest."""
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    shards = sorted(input_dir.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"No *.safetensors files in {input_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"{output_dir} is not empty; remove it or pick another --output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_size = 0
    for shard in shards:
        total_size += convert_shard(shard, output_dir / shard.name)
        print(f"Converted {shard.name}")

    # Other files of the checkpoint, then the tokenizer (which wins on a name clash)
    sources = [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix != ".safetensors"]
    if args.tokenizer_dir:
        sources += [p for p in sorted(Path(args.tokenizer_dir).iterdir()) if p.is_file()]
    for path in sources:
        match path.name:
            case "config.json":
                write_config(path, output_dir / path.name)
            case "model.safetensors.index.json":
                # The index maps tensors to shard files, which keep their names; only the size changes
                index = json.loads(path.read_text())
                index.setdefault("metadata", {})["total_size"] = total_size
                (output_dir / path.name).write_text(json.dumps(index, indent=2) + "\n")
            case _:
                shutil.copy2(path, output_dir / path.name)
    if not (output_dir / "config.json").exists():
        print(f"Warning: no config.json in {input_dir}")
    print(f"Wrote the bf16 checkpoint to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write a bf16 copy of an fp32 HuggingFace checkpoint dir.")
    parser.add_argument("--input_dir", required=True, help="Dir with the fp32 *.safetensors and config.json.")
    parser.add_argument("--output_dir", required=True, help="Dir for the bf16 copy (must be empty or new).")
    parser.add_argument(
        "--tokenizer_dir",
        default=None,
        help="Dir whose files (the tokenizer) are copied into the output too.",
    )
    args = parser.parse_args()
    main(args)
