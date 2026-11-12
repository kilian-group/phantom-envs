"""Tests for scripts/convert_hf_fp32_to_bf16.py on a tiny fake checkpoint."""

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

# torch and safetensors come with the skyrl extra
pytest.importorskip("torch")
pytest.importorskip("safetensors")

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert_hf_fp32_to_bf16.py"
spec = importlib.util.spec_from_file_location("convert_hf_fp32_to_bf16", SCRIPT)
convert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert)


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    """An fp32 checkpoint dir with two shards, an index, a config and a tokenizer file."""
    ckpt = tmp_path / "fp32"
    ckpt.mkdir()
    save_file({"a": torch.randn(4, 4), "ids": torch.arange(3)}, ckpt / "model-00001-of-00002.safetensors")
    save_file({"b": torch.randn(2)}, ckpt / "model-00002-of-00002.safetensors")
    index = {
        "metadata": {"total_size": 72},
        "weight_map": {
            "a": "model-00001-of-00002.safetensors",
            "ids": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors",
        },
    }
    (ckpt / "model.safetensors.index.json").write_text(json.dumps(index))
    (ckpt / "config.json").write_text(json.dumps({"dtype": "float32", "hidden_size": 4}))
    (ckpt / "generation_config.json").write_text("{}")
    return ckpt


def run(input_dir: Path, output_dir: Path, tokenizer_dir: Path | None = None) -> None:
    convert.main(argparse.Namespace(input_dir=input_dir, output_dir=output_dir, tokenizer_dir=tokenizer_dir))


def test_converts_floats_and_keeps_the_rest(checkpoint, tmp_path):
    out = tmp_path / "bf16"
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")
    run(checkpoint, out, tokenizer)

    shard = load_file(out / "model-00001-of-00002.safetensors")
    original = load_file(checkpoint / "model-00001-of-00002.safetensors")
    assert shard["a"].dtype == torch.bfloat16
    assert torch.equal(shard["a"], original["a"].to(torch.bfloat16))
    assert shard["ids"].dtype == torch.int64  # integer tensors stay as they are
    assert load_file(out / "model-00002-of-00002.safetensors")["b"].dtype == torch.bfloat16

    config = json.loads((out / "config.json").read_text())
    assert config == {"dtype": "bfloat16", "hidden_size": 4}
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == 16 * 2 + 3 * 8 + 2 * 2
    assert index["weight_map"]["b"] == "model-00002-of-00002.safetensors"
    assert (out / "generation_config.json").exists()
    assert (out / "tokenizer.json").exists()


def test_refuses_a_non_empty_output_dir(checkpoint, tmp_path):
    out = tmp_path / "bf16"
    out.mkdir()
    (out / "keep.txt").write_text("")
    with pytest.raises(SystemExit):
        run(checkpoint, out)
