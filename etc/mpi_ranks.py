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
        # No mpi-pytest here. That is fine for the many trees with no MPI tests, but if something
        # *does* carry a `parallel` marker then pytest has quietly ignored it and would run that
        # test on one rank -- passing while testing nothing, or failing for a puzzling reason.
        declared = [item.nodeid for item in items if any(m.name == 'parallel' for m in item.own_markers)]
        if declared:
            pytest.exit(
                f"{len(declared)} test(s) carry @pytest.mark.parallel but mpi-pytest is not installed "
                f"in this environment, so the marker is ignored and they would run on one rank. "
                f"Add mpi-pytest to it. First: {declared[0]}",
                returncode=4,
            )
        return

    for item in items:
        if item.get_closest_marker("parallel"):
            _ranks.add(int(_extract_nprocs_for_single_test(item)))


def pytest_collection_finish(session):
    print("MPI_RANKS " + " ".join(str(n) for n in sorted(_ranks)))
