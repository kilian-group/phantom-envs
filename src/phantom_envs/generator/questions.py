"""Sample questions from every template and answer them from the database.

Each template restarts the random generator from the seed, so its questions do not depend on
which other templates are generated.
"""

import logging

from numpy.random import default_rng
from tqdm import tqdm

from phantom_envs._types import EnvConfig
from phantom_envs.generator.bidirectional import sample_bidirectional
from phantom_envs.generator.difficulty import compute_difficulty
from phantom_envs.generator.query import answer_query
from phantom_envs.generator.relations import Database
from phantom_envs.generator.sample import Lookups, sample_backward
from phantom_envs.generator.templates import Template, generate_templates
from phantom_envs.generator.vocab import relation_bank

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
        The older of the two (the younger for "Who is younger"); a tie goes to the second.
    """
    left, right = (bindings[v] for v in template.answer)
    left_dob, right_dob = (db["dob"].of(person)[0] for person in (left, right))
    if template.question[0] == "Who is younger,":
        return [left] if left_dob > right_dob else [right]
    return [left] if left_dob < right_dob else [right]


def sample_questions(db: Database, config: EnvConfig) -> list[dict]:
    """Sample and answer the questions of all of a config's templates.

    Args:
        db: The universe.
        config: The environment config.

    Returns:
        In template order, dicts with ``question``, ``answers``, ``kind``, ``template`` (index into the config's
        templates), ``prolog`` (``query`` and ``answer`` variable) and ``difficulty``.
    """
    templates = samplable_templates(config)
    lookups = Lookups(db, relation_bank(config.hard_mode))
    samples = [
        (i, sample)
        for i, template in enumerate(tqdm(templates, desc="Sampling questions"))
        for sample in sample_template(template, config, lookups)
    ]

    questions = []
    for i, (question, query, bindings) in tqdm(samples, desc="Answering questions"):
        template = templates[i]
        if template.kind == "comparison":
            # The sample already names the two people to compare
            answers = comparison_answer(template, bindings, db)
        else:
            answers = answer_query(db, query, template.answer)
        answer_var = str(template.answer)
        questions.append(
            {
                "question": question,
                "answers": answers,
                "kind": template.kind,
                "template": i,
                "prolog": {"query": query, "answer": answer_var},
                "difficulty": compute_difficulty(query, answer_var=answer_var),
            }
        )
    return questions
