"""Dense encoder presets and the on-disk layout of a retrieval corpus.

Every data dir holds ``corpus.jsonl`` (one ``{"id", "title"?, "contents"}`` object per line) and one
flat FAISS index per encoder, ``<encoder>_Flat.index``.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


def load_encoder(name: str):
    """Load encoder ``name`` as a SentenceTransformer."""
    from sentence_transformers import SentenceTransformer

    preset = get_encoder(name)
    return SentenceTransformer(preset.model, **preset.load_kwargs)


def index_path(data_dir: str | Path, encoder: str) -> Path:
    """Path of the flat FAISS index for ``encoder`` in ``data_dir``."""
    return Path(data_dir) / f"{encoder}_Flat.index"


def corpus_path(data_dir: str | Path) -> Path:
    """Path of the corpus in ``data_dir``."""
    return Path(data_dir) / CORPUS_FILENAME
