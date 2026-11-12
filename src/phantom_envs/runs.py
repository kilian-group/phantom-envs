"""Names of eval outputs, shared by the eval scripts and the analysis scripts that read them.

An eval of a model writes to ``<out_dir>/model=<compact_model_slug(model_path)>/``.
"""

from pathlib import Path


def slugify_path(path: str) -> str:
    """Replace path separators with double dashes for use in directory names.

    Strips a trailing ``.parquet`` suffix when present so eval output dirs don't end in
    literal ``.parquet``.

    Examples:
        ``Qwen/Qwen2.5-3B-Instruct`` → ``Qwen--Qwen2.5-3B-Instruct``
    """
    p = Path(path)
    if p.suffix == ".parquet":
        path = str(p.with_suffix(""))
    return path.replace("/", "--")


def compact_model_slug(model_path: str) -> str:
    """Filesystem-safe slug for a model path.

    Collapses the training-export layout ``.../<run_name>/exports/global_step_<N>/policy``
    (optionally with ``/huggingface`` appended) to ``<run_name>__step<N>``; otherwise slugifies
    the full path. Without this, checkpoint slugs blow past Linux's 255-byte filename limit.
    """
    parts = Path(model_path).parts
    if parts and parts[0] == "/":  # absolute path: drop the root, which would slugify to a leading "--"
        parts = parts[1:]
    if parts and parts[-1] == "huggingface":
        parts = parts[:-1]
    if (
        len(parts) >= 4
        and parts[-1] == "policy"
        and parts[-3] == "exports"
        and parts[-2].startswith("global_step_")
    ):
        run_name = "/".join(parts[:-3])
        step = parts[-2].removeprefix("global_step_")
        path = f"{run_name}__step{step}"
    else:
        path = "/".join(parts)
    return slugify_path(path)
