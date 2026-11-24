"""Run a function over many items in worker processes.

The workers are forked, so they see everything the parent process has in memory at that moment (the
universe's database, the articles, ...) without it being copied or sent to them. Only the items and
the results travel between processes.
"""

import multiprocessing
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

# The function a worker applies. Set in each worker as it starts, never in the parent process.
_function: Callable | None = None


class ProgressBar(tqdm):
    """A progress bar without tqdm's monitor thread, since forking a process that has threads can deadlock."""

    monitor_interval = 0


def _set_function(function: Callable) -> None:
    """Remember the function to apply (runs once in each worker)."""
    global _function
    _function = function


def _apply[Item, Result](item: Item) -> Result:
    """Apply the worker's function to one item (runs in a worker)."""
    return _function(item)


def parallel_map[Item, Result](
    function: Callable[[Item], Result], items: Sequence[Item], num_workers: int, desc: str | None = None
) -> list[Result]:
    """Apply a function to every item, in forked worker processes.

    The result does not depend on the number of workers, as long as the function's result depends
    only on its item (and on data that existed before the call).

    Args:
        function: Any callable, including one that holds large data: workers inherit it by fork.
        items: The items. They and the results are sent between processes, so keep them small.
        num_workers: Number of worker processes; 1 runs in this process.
        desc: Label of a progress bar, or None for no bar.

    Returns:
        The function's result for each item, in the order of `items`.

    Raises:
        concurrent.futures.process.BrokenProcessPool: If a worker dies (e.g. out of memory).
        Exception: Whatever the function raises for an item.
    """
    num_workers = min(num_workers, len(items))
    # Without fork (Windows), or inside a worker that may not start its own, run in this process
    can_fork = "fork" in multiprocessing.get_all_start_methods()
    if num_workers <= 1 or not can_fork or multiprocessing.current_process().daemon:
        return [function(item) for item in ProgressBar(items, desc=desc, disable=desc is None)]

    fork = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(
        num_workers, mp_context=fork, initializer=_set_function, initargs=(function,)
    ) as pool:
        # Several chunks per worker, so that one slow chunk does not leave the others idle
        chunksize = max(1, len(items) // (num_workers * 16))
        results = pool.map(_apply, items, chunksize=chunksize)
        return list(ProgressBar(results, total=len(items), desc=desc, disable=desc is None))
