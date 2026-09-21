"""
Report the rank counts the collected tests ask for.

Loaded with ``-p mpi_ranks`` alongside ``--collect-only``, this prints one ``MPI_RANKS ...`` line
listing the distinct ``@pytest.mark.parallel(n)`` values in the selection. That lets the runner take
the rank counts from the tests themselves instead of a list kept in the workflow, so the two cannot
drift apart.
"""

import pytest

_ranks = set()


# `trylast`, so this sees the selection after pytest's own `-m` deselection rather than every
# collected item.
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    try:
        from pytest_mpi.plugin import _extract_nprocs_for_single_test
    except ImportError:
        # No mpi-pytest in this environment, so nothing can carry a `parallel` marker and there are
        # no rank counts to report. Most project environments are in this position.
        return

    for item in items:
        if item.get_closest_marker("parallel"):
            _ranks.add(int(_extract_nprocs_for_single_test(item)))


def pytest_collection_finish(session):
    print("MPI_RANKS " + " ".join(str(n) for n in sorted(_ranks)))
