"""
Refuse tests that no CI job would run.

CI runs this tree once per environment, as ``pytest pySDC/tests -m <marker>`` (the
``user_cpu_tests_linux`` matrix and the firedrake job in ``.github/workflows/ci_pipeline.yml``, and
``etc/modal_gpu_tests.py`` for ``cupy``). A test carrying none of those markers is deselected by every
one of them, so it is never run anywhere, silently. Eight tests had drifted into that state.
"""

from pathlib import Path

import pytest

# the markers the CI jobs select for this tree
CI_MARKERS = {'base', 'fenics', 'mpi4py', 'petsc', 'pytorch', 'firedrake', 'cupy'}
# deliberately not run by CI; run them with `-m benchmark`
NOT_IN_CI = {'benchmark'}

HERE = Path(__file__).parent


# `tryfirst`, so that this sees every collected test and not only what survives `-m`: a CI run with
# `-m base` has to fail on an unmarked test rather than quietly deselect it.
@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    unmarked = [
        item.nodeid
        for item in items
        if HERE in Path(item.path).parents and not {m.name for m in item.iter_markers()} & (CI_MARKERS | NOT_IN_CI)
    ]
    if unmarked:
        pytest.exit(
            f'{len(unmarked)} test(s) carry none of the markers {sorted(CI_MARKERS)}, so no CI job runs them. '
            f'Mark them, or `benchmark` if they are deliberately left out of CI. First: {unmarked[0]}',
            returncode=4,
        )
