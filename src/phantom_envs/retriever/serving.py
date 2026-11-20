"""What the retrieval servers share: their config, snippets, and how they bind a port.

Two servers put a retriever behind a wire protocol: rest_server.py (JSON over HTTP, for the search
env) and mcp_server.py (the Model Context Protocol, for the terminal agent's search tool).
"""

import argparse
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

from omegaconf import OmegaConf

from phantom_envs.retriever.base import RETRIEVERS
from phantom_envs.retriever.encoders import ENCODERS, get_encoder
from phantom_envs.retriever.snippets import SnippetCutter


@dataclass
class RetrieverServerConfig:
    """Retrieval server settings; retriever and data_dir locate the index and corpus."""

    retriever: str = field(metadata={"description": f"Retriever: one of {list(RETRIEVERS)}."})
    data_dir: str = field(metadata={"description": "Dir with corpus.jsonl and the retriever's index."})
    topk: int = field(default=3, metadata={"description": "Default number of documents per query."})
    faiss_gpu: bool = field(
        default=False,
        metadata={"description": "Dense only: shard the index across all GPUs in fp16 (needs faiss-gpu)."},
    )
    device: str | None = field(
        default=None,
        metadata={
            "description": "Dense only: where the encoder runs, e.g. cuda:1, or cpu for e5 to leave the "
            "GPUs to a trainer. Default: the encoder preset's choice, or a GPU if there is one."
        },
    )
    port: int = field(default=8000, metadata={"description": "HTTP port; 0 lets the kernel pick a free one."})
    port_file: str | None = field(
        default=None, metadata={"description": "If set, the bound port is written here."}
    )
    snippet_max_tokens: int = field(
        default=-1,
        metadata={
            "description": "Truncate each returned document to its first N tokens (-1 = full document)."
        },
    )
    snippet_tokenizer: str | None = field(
        default=None,
        metadata={
            "description": "HuggingFace model whose tokenizer counts a snippet's tokens. Default: the "
            "dense encoder's; bm25 has none, so it needs this with snippet_max_tokens."
        },
    )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> Self:
        """Build the config from the command line.

        Args:
            args: The parsed --retriever and --data_dir, with the remaining key=value arguments in
                args.overrides.

        Returns:
            The config, with the overrides applied.
        """
        base = OmegaConf.structured(cls(retriever=args.retriever, data_dir=args.data_dir))
        return cls(**OmegaConf.to_container(OmegaConf.merge(base, OmegaConf.from_cli(args.overrides))))


def parse_args(description: str) -> argparse.Namespace:
    """Parse a retrieval server's command line.

    Args:
        description: What the server is, for --help.

    Returns:
        The parsed arguments, for RetrieverServerConfig.from_args.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--retriever", required=True, choices=list(RETRIEVERS), help="Retriever to serve.")
    parser.add_argument("--data_dir", required=True, help="Dir with corpus.jsonl and the retriever's index.")
    args, args.overrides = parser.parse_known_args()
    return args


def snippet_cutter(cfg: RetrieverServerConfig) -> SnippetCutter | None:
    """Return what cuts the server's documents to snippets, if the config asks for snippets.

    Args:
        cfg: The server's config.

    Returns:
        The cutter, or None to serve full documents.

    Raises:
        ValueError: If snippets are on and no tokenizer is named or implied by the retriever.
    """
    if cfg.snippet_max_tokens <= 0:
        return None
    tokenizer = cfg.snippet_tokenizer
    if tokenizer is None and cfg.retriever in ENCODERS:
        tokenizer = get_encoder(cfg.retriever).model
    if tokenizer is None:
        raise ValueError(
            f"snippet_max_tokens needs snippet_tokenizer=<HuggingFace model> with retriever {cfg.retriever!r}"
        )
    return SnippetCutter(tokenizer, cfg.snippet_max_tokens)


def bind_port(cfg: RetrieverServerConfig, host: str) -> socket.socket:
    """Bind the server's port, and report it in the config's port file.

    Call this before loading the retriever, which can take minutes, so the port is known right
    away; the server starts listening on the socket once it is ready.

    Args:
        cfg: The server's config.
        host: The address to bind.

    Returns:
        The bound socket.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, cfg.port))
    if cfg.port_file:
        Path(cfg.port_file).write_text(f"{sock.getsockname()[1]}\n")
    return sock
