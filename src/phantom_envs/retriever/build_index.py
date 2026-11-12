#!/usr/bin/env -S uv run python
"""Build a FAISS index over the corpus in a data dir.

Encodes ``<data_dir>/corpus.jsonl`` with a dense encoder preset (``e5`` = e5-base-v2 by default,
``qwen3-4b`` = Qwen3-Embedding-4B; see :mod:`phantom_envs.retriever.encoders`) via
sentence-transformers and builds an exact (flat) FAISS inner-product index. FAISS is not a
dependency of the venv, so run it with ``uv run --with faiss-cpu==1.12.0``.

Usage:
    uv run --with faiss-cpu==1.12.0 python -m phantom_envs.retriever.build_index \
        --data_dir data/pw/depth_16_size_1000_seed_1

    # Qwen3-Embedding-4B (bf16)
    uv run --with faiss-cpu==1.12.0 python -m phantom_envs.retriever.build_index \
        --data_dir data/hp500 --encoder qwen3-4b

Output (written into data_dir):
    {encoder}_Flat.index       — FAISS IndexFlatIP over normalized embeddings
    {encoder}_embeddings.mmap  — the embeddings, kept so an interrupted run resumes
"""

import argparse
import logging
from pathlib import Path

import datasets
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from phantom_envs.retriever.encoders import ENCODERS, corpus_path, get_encoder, index_path, load_encoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_corpus(corpus_path: Path) -> list[str]:
    """Load document texts from a corpus JSONL file.

    Uses HuggingFace datasets to handle both plain-text and binary-serialized JSONL.

    Args:
        corpus_path: Path to the corpus JSONL file (e.g. wiki-18.jsonl).

    Returns:
        List of document text strings (one per document).
    """
    logger.info(f"Loading existing corpus from {corpus_path} ...")
    ds = datasets.load_dataset("json", data_files=str(corpus_path), split="train", num_proc=4)
    logger.info(f"Loaded {len(ds)} documents from {corpus_path}")
    return ds["contents"]


def encode_to_memmap(
    texts: list[str],
    model: SentenceTransformer,
    encoder: str,
    batch_size: int,
    chunk_size: int,
    memmap_path: Path,
) -> tuple[np.memmap, int, int]:
    """Encode texts in chunks and write embeddings to a memory-mapped file.

    This avoids holding all embeddings in RAM at once, which is critical for
    large corpora (e.g. 21M Wikipedia documents × 3584 dim ≈ 280 GB).

    Args:
        texts: Raw document texts.
        model: Loaded SentenceTransformer model.
        encoder: Encoder preset name (selects the document encode kwargs).
        batch_size: Encoding batch size (passed to model.encode).
        chunk_size: Number of documents to encode per chunk before flushing to disk.
        memmap_path: File path for the memory-mapped embedding file.

    Returns:
        Tuple of (memmap array, num_documents, embedding_dim).
    """
    n = len(texts)
    encode_kwargs = get_encoder(encoder).doc_kwargs

    # Probe embedding dimension with a single text
    probe = model.encode(["hello"], normalize_embeddings=True, **encode_kwargs)
    dim = probe.shape[1]
    logger.info(f"Embedding dimension: {dim}, documents: {n}")

    # Resume support: reuse an existing memmap if it has the right shape
    expected_bytes = n * dim * np.dtype("float32").itemsize
    if memmap_path.exists() and memmap_path.stat().st_size == expected_bytes:
        logger.info(f"Found existing memmap at {memmap_path}, resuming ...")
        memmap = np.memmap(str(memmap_path), dtype="float32", mode="r+", shape=(n, dim))
    else:
        memmap = np.memmap(str(memmap_path), dtype="float32", mode="w+", shape=(n, dim))

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        # Skip chunks that are already encoded (any non-zero row means the chunk was written)
        if np.any(memmap[start]):
            logger.info(f"Skipping chunk [{start}:{end}] / {n} (already encoded)")
            continue

        logger.info(f"Encoding chunk [{start}:{end}] / {n} ...")
        chunk_emb = model.encode(
            texts[start:end],
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=True,
            **encode_kwargs,
        )
        memmap[start:end] = chunk_emb
        memmap.flush()

    logger.info(f"Embeddings written to {memmap_path} — shape ({n}, {dim})")
    return memmap, n, dim


def build_index(
    embeddings: np.ndarray | np.memmap,
    n: int,
    dim: int,
    add_chunk_size: int = 100_000,
) -> faiss.IndexFlatIP:
    """Build a flat FAISS inner-product index from pre-computed embeddings.

    For large corpora backed by a memmap, vectors are added in chunks to avoid
    loading the entire embedding matrix into RAM.

    Args:
        embeddings: Embedding array of shape (n, dim), may be a memmap.
        n: Number of vectors.
        dim: Embedding dimension.
        add_chunk_size: Number of vectors to add to the index per batch.

    Returns:
        Populated FAISS index.
    """
    index = faiss.IndexFlatIP(dim)

    # Add vectors in chunks to keep RAM bounded
    for start in range(0, n, add_chunk_size):
        end = min(start + add_chunk_size, n)
        logger.info(f"Adding vectors [{start}:{end}] / {n} to index ...")
        index.add(np.array(embeddings[start:end]))

    logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def main(args: argparse.Namespace) -> None:
    """Encode the data dir's corpus and write its flat FAISS index."""
    data_dir = Path(args.data_dir).expanduser()
    texts = load_corpus(corpus_path(data_dir))

    # Load encoder
    logger.info(f"Loading encoder: {get_encoder(args.encoder).model}")
    model = load_encoder(args.encoder)
    model.max_seq_length = args.max_length

    # Encode to memmap
    memmap_path = data_dir / f"{args.encoder}_embeddings.mmap"
    embeddings, n, dim = encode_to_memmap(
        texts,
        model,
        encoder=args.encoder,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        memmap_path=memmap_path,
    )

    # Build index
    index = build_index(embeddings, n, dim)

    # Save
    out_path = index_path(data_dir, args.encoder)
    faiss.write_index(index, str(out_path))
    logger.info(f"Index saved to {out_path}")

    # DO NOT clean up memmap file, for future reuse!
    # memmap_path.unlink()
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a FAISS index over a data dir's corpus.jsonl.")
    parser.add_argument(
        "--data_dir", required=True, help="Directory with corpus.jsonl; the index is written here."
    )
    parser.add_argument(
        "--encoder",
        default="e5",
        choices=list(ENCODERS),
        help="Encoder preset; also names the index file <encoder>_Flat.index (default: e5).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Encoding batch size (default: 64).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum token length for encoding (default: 512).",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100_000,
        help="Number of documents to encode per chunk before flushing to disk (default: 100000).",
    )
    args = parser.parse_args()
    main(args)
