"""Split an environment's tasks into train and validation, balanced across cells.

With the paper's settings this reproduces its splits exactly: pw stratified by difficulty,
pwcons with every (hops, constraints) cell cut to the smallest, and pwcomp with every (hops, kind)
cell capped at 4x the smallest.
"""

import random
from collections import defaultdict

from phantom_envs._types import SplitConfig, Task


def split_tasks(tasks: list[Task], config: SplitConfig) -> tuple[list[Task], list[Task]]:
    """Split tasks into train and validation, cell by cell.

    Each cell is shuffled and capped, then ``max(1, int(len * val_ratio))`` of it goes to
    validation. The order of random draws matches the paper's prepare scripts.

    Args:
        tasks: All tasks, in generation order.
        config: The cells, cap, ratio and seed.

    Returns:
        The train and validation tasks, each shuffled.
    """
    rng = random.Random(config.seed)
    # Task indices per cell, in task order
    cells: dict[tuple, list[int]] = defaultdict(list)
    for idx, task in enumerate(tasks):
        cells[tuple(task.metadata[key] for key in config.cells)].append(idx)

    cap = None if config.cap_factor is None else config.cap_factor * min(len(c) for c in cells.values())
    # Cells in sorted order, so the shuffles draw from the generator in a fixed order
    train, val = [], []
    for key in sorted(cells):
        indices = cells[key]
        rng.shuffle(indices)
        indices = indices[:cap]
        num_val = max(1, int(len(indices) * config.val_ratio))
        val += indices[:num_val]
        train += indices[num_val:]
    rng.shuffle(train)
    rng.shuffle(val)
    return [tasks[i] for i in train], [tasks[i] for i in val]
