"""Shared types: environment configs and tasks, and the retrieval API's request and response models.

The retrieval models are used by both the retrieval server (retriever/retrieval_server.py) and the
search client (retriever/client.py), so neither imports the other.
"""

from dataclasses import dataclass, field

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class SplitConfig:
    """How an environment's questions become train and validation tasks.

    Questions are grouped into cells by ``cells``; each cell is shuffled, capped at ``cap_factor``
    times the smallest cell, and split by ``val_ratio``.
    """

    val_ratio: float = field(default=0.1, metadata={"description": "Share of each cell for validation."})
    seed: int = field(default=1, metadata={"description": "Seed for the shuffles."})
    cells: tuple[str, ...] = field(
        default=("hops", "constraints"), metadata={"description": "Task metadata fields that define a cell."}
    )
    cap_factor: int | None = field(
        default=None,
        metadata={"description": "Cap per cell, in multiples of the smallest cell (None: no cap)."},
    )


@dataclass(frozen=True)
class EnvConfig:
    """A PhantomEnvs environment: a fictional universe and the questions asked about it."""

    name: str = field(
        default="pw", metadata={"description": "The tasks' data_source (names the env in run paths)."}
    )
    size: int = field(default=1000, metadata={"description": "Target number of people (a multiple of 10)."})
    seed: int = field(default=1, metadata={"description": "Seed for the universe and the questions."})
    max_hops: int = field(default=6, metadata={"description": "Most relation hops in a question."})
    max_constraints: int = field(
        default=1, metadata={"description": "Most attribute constraints in a question."}
    )
    kinds: tuple[str, ...] = field(
        default=("base",),
        metadata={
            "description": "Question kinds: base (who / what), count (how many), comparison (who is older)."
        },
    )
    sampler: str = field(
        default="bidirectional",
        metadata={
            "description": "backward (walk from the chain's far end) or bidirectional (from a random anchor)."
        },
    )
    hard_mode: bool = field(
        default=False,
        metadata={
            "description": "Also use relations that take several articles to resolve (aunt, cousin, ...)."
        },
    )
    questions_per_template: int = field(
        default=1000,
        metadata={"description": "Questions sampled per template; also its sampling-attempt budget."},
    )
    split: SplitConfig = field(
        default_factory=SplitConfig, metadata={"description": "Train / validation split."}
    )


@dataclass
class Task:
    """One question with its answers: the unit every data dir stores in ``tasks/<split>.jsonl``."""

    id: str = field(metadata={"description": "Stable id (PhantomEnvs: hash of question and answers)."})
    data_source: str = field(
        metadata={"description": "Dataset the task comes from (pw, pwcons, hp500, searchR1_nq, ...)."}
    )
    question: str = field(metadata={"description": "The question."})
    answers: list[str] = field(metadata={"description": "All correct answers."})
    metadata: dict = field(
        default_factory=dict,
        metadata={
            "description": "Source-specific fields. PhantomEnvs: kind, hops, constraints, difficulty, "
            "template, prolog. Benchmarks: their native fields, passed on to trainer formats as is."
        },
    )


class Document(BaseModel):
    """A single document from the corpus."""

    id: str | None = Field(default=None, description="Unique document identifier.")
    title: str | None = Field(default=None, description="Document title.")
    contents: str = Field(description="Document text.")

    def __str__(self) -> str:
        if self.title is not None:
            return f"Title: {self.title}\nContents: {self.contents.strip()}"
        return f"Contents: {self.contents.strip()}"


class DocumentResult(BaseModel):
    """One retrieved document together with its optional retrieval score."""

    document: Document = Field(description="The retrieved Wikipedia passage.")
    score: float | None = Field(
        default=None,
        description="FAISS inner-product score; None when scores are not requested.",
    )


class RetrievalResponse(BaseModel):
    """Response body returned by POST /retrieve.

    ``result`` is a flat list of the top-k retrieved documents (length <= topk).
    """

    result: list[DocumentResult] = Field(
        description="Top-k retrieved documents, ordered by descending score.",
    )

    def __str__(self) -> str:
        return "".join(f"Doc {idx + 1}:\n{str(r.document)}\n\n" for idx, r in enumerate(self.result))
