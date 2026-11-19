"""Dense encoder presets and the on-disk layout of a retrieval corpus.

Every data dir holds ``corpus.jsonl`` (one ``{"id", "title"?, "contents"}`` object per line) and one
flat FAISS index per encoder, ``<encoder>_Flat.index``.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

CORPUS_FILENAME = "corpus.jsonl"


@dataclass(frozen=True)
class EncoderPreset:
    """How to load one dense encoder and encode queries and documents with it."""

    model: str = field(metadata={"description": "HuggingFace model id."})
    query_kwargs: dict[str, Any] = field(
        default_factory=dict, metadata={"description": "SentenceTransformer.encode kwargs for queries."}
    )
    doc_kwargs: dict[str, Any] = field(
        default_factory=dict, metadata={"description": "SentenceTransformer.encode kwargs for documents."}
    )
    load_kwargs: dict[str, Any] = field(
        default_factory=dict, metadata={"description": "SentenceTransformer constructor kwargs."}
    )


ENCODERS: dict[str, EncoderPreset] = {
    # Training retriever: e5 expects "query: " / "passage: " prefixes.
    "e5": EncoderPreset(
        model="intfloat/e5-base-v2",
        query_kwargs={"prompt": "query"},
        doc_kwargs={"prompt": "passage"},
    ),
    # Eval retriever: Qwen3-Embedding's built-in query prompt; documents take no prompt.
    "qwen3-4b": EncoderPreset(
        model="Qwen/Qwen3-Embedding-4B",
        query_kwargs={"prompt_name": "query"},
        load_kwargs={
            "model_kwargs": {
                "dtype": "bfloat16",
                "attn_implementation": "flash_attention_2",
                "device_map": "cuda:0",  # Load only on the first GPU
            },
            "tokenizer_kwargs": {"padding_side": "left"},
        },
    ),
}


def get_encoder(name: str) -> EncoderPreset:
    """Return the preset for encoder ``name``.

    Raises:
        ValueError: If ``name`` is not a known encoder.
    """
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder {name!r}; choose from {list(ENCODERS)}")
    return ENCODERS[name]


def load_encoder(name: str, device: str | None = None) -> "SentenceTransformer":
    """Load an encoder preset as a SentenceTransformer.

    Args:
        name: The preset.
        device: Where to run it, e.g. cpu or cuda:1; it replaces a device the preset loads on. None
            keeps the preset's own choice, or a GPU if there is one.

    Returns:
        The encoder.
    """
    from sentence_transformers import SentenceTransformer

    preset = get_encoder(name)
    load_kwargs = dict(preset.load_kwargs)
    if device is not None and "device_map" in load_kwargs.get("model_kwargs", {}):
        load_kwargs["model_kwargs"] = {**load_kwargs["model_kwargs"], "device_map": device}
    return SentenceTransformer(preset.model, device=device, **load_kwargs)


def index_path(data_dir: str | Path, encoder: str) -> Path:
    """Path of the flat FAISS index for ``encoder`` in ``data_dir``."""
    return Path(data_dir) / f"{encoder}_Flat.index"


def corpus_path(data_dir: str | Path) -> Path:
    """Path of the corpus in ``data_dir``."""
    return Path(data_dir) / CORPUS_FILENAME
