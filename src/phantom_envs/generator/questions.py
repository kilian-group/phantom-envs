"""Sample questions from every template and answer them from the database.

Each template restarts the random generator from the seed, so its questions do not depend on
which other templates are generated.
"""

import logging

from numpy.random import default_rng

from phantom_envs._types import EnvConfig
from phantom_envs.generator.bidirectional import sample_bidirectional
from phantom_envs.generator.difficulty import compute_difficulty
from phantom_envs.generator.parallel import parallel_map
from phantom_envs.generator.query import answer_query
from phantom_envs.generator.relations import Database
from phantom_envs.generator.sample import Lookups, sample_backward
from phantom_envs.generator.templates import Template, generate_templates
from phantom_envs.generator.vocab import ATTRIBUTE_TYPES, relation_bank

logger = logging.getLogger(__name__)


def samplable_templates(config: EnvConfig) -> list[Template]:
    """The templates a config samples from.

    Args:
        config: The environment config.

    Returns:
        Its templates, without comparisons of a bare ``<name>``: the sampler cannot fill those
        (PhantomWiki generates none of their questions either).
    """
    templates = generate_templates(config.max_hops, config.max_constraints, config.kinds)
    return [
        t for t in templates if not (t.kind == "comparison" and any(a.startswith("<name>") for a in t.query))
    ]


def sample_template(
    template: Template, config: EnvConfig, lookups: Lookups
) -> list[tuple[str, list[str], dict]]:
    """Sample questions from one template.

    Args:
        template: The template.
        config: The environment config (seed, sampler, questions per template).
        lookups: The universe's lookups.

    Returns:
        Up to ``questions_per_template`` (question, query, bindings) samples; bindings are only
        filled for comparisons. Duplicates are kept.
    """
    # Each template restarts from the seed, so its questions do not depend on the other templates
    rng = default_rng(config.seed)
    n = config.questions_per_template
    samples = []
    attempts = 0
    while len(samples) < n and attempts < n * n:
        attempts += 1
        if template.kind == "comparison":
            # One walk per attempt; some comparisons cannot be walked at all
            try:
                result = sample_backward(template, rng, lookups, attempts=1)
            except (ValueError, AssertionError):
                continue
        elif config.sampler == "bidirectional" and (found := sample_bidirectional(template, rng, lookups, n)):
            result = (*found, {})
        else:
            # Also the fallback when no bidirectional walk works
            result = sample_backward(template, rng, lookups, attempts=n)
        if result is not None:
            samples.append(result)
    if len(samples) < n:
        logger.info(f"Only {len(samples)}/{n} questions for template {' '.join(template.question)}")
    return samples


def comparison_answer(template: Template, bindings: dict[str, str], db: Database) -> list[str]:
    """Answer a comparison from the two people its sample resolved to.

    Args:
        template: The comparison template; its answer holds the two people's variables.
        bindings: Placeholder -> person, from the sample.
        db: The universe (for dates of birth).

    Returns:
        The person the question asks for; a tie goes to the second.

    Raises:
        ValueError: If the question is not one of the comparison phrasings.
    """
    left, right = (bindings[v] for v in template.answer)
    left_dob, right_dob = (db["dob"].of(person)[0] for person in (left, right))
    match template.question[0]:
        case "Who is older," | "Who was born first,":
            return [left] if left_dob < right_dob else [right]
        case "Who is younger,":
            return [left] if left_dob > right_dob else [right]
        case phrasing:
            raise ValueError(f"Cannot answer the comparison {phrasing!r}")


def answer_sample(
    db: Database, templates: list[Template], index: int, sample: tuple[str, list[str], dict]
) -> dict:
    """Answer one sampled question and score its difficulty.

    Args:
        db: The universe.
        templates: The config's templates.
        index: The template the sample comes from.
        sample: The question, its query and its bindings, from `sample_template`.

    Returns:
        A dict with `question`, `answers`, `kind`, `template`, `prolog` (`query` and `answer` variable)
        and `difficulty`.
    """
    template = templates[index]
    question, query, bindings = sample
    if template.kind == "comparison":
        # The sample already names the two people to compare
        answers = comparison_answer(template, bindings, db)
    else:
        answers = answer_query(db, query, template.answer)
    answer_var = str(template.answer)
    return {
        "question": question,
        "answers": answers,
        "kind": template.kind,
        "template": index,
        "prolog": {"query": query, "answer": answer_var},
        "difficulty": compute_difficulty(query, answer_var=answer_var),
    }


def sample_questions(db: Database, config: EnvConfig, num_workers: int = 1) -> list[dict]:
    """Sample and answer the questions of all of a config's templates.

    Args:
        db: The universe.
        config: The environment config.
        num_workers: Processes to sample and answer with; the questions do not depend on it.

    Returns:
        The questions (see `answer_sample`), in template order.
    """
    templates = samplable_templates(config)
    lookups = Lookups(db, relation_bank(config.hard_mode))
    db.derive(lookups.relations + ATTRIBUTE_TYPES)

    # A template's questions do not depend on the other templates, nor an answer on other questions,
    # so workers can share out both
    sampled = parallel_map(
        lambda template: sample_template(template, config, lookups),
        templates,
        num_workers,
        desc="Sampling questions",
    )
    samples = [(index, sample) for index, of_template in enumerate(sampled) for sample in of_template]
    return parallel_map(
        lambda indexed: answer_sample(db, templates, *indexed),
        samples,
        num_workers,
        desc="Answering questions",
    )
