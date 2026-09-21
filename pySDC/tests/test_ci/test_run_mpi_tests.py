"""
Checks for `etc/run_mpi_tests.sh` and its rank-discovery plugin.

This script decides what the whole MPI test suite runs, and its failure mode is a job that passes
having launched less than it should -- which looks exactly like success. Two such bugs have already
been caught by hand (an empty bash array under `set -u`, and selecting an unknown marker), so the
launches it makes are asserted here rather than trusted.

The passes are stubbed: `PYTEST` and `mpiexec` are replaced with scripts that append their argv to a
log and exit with a chosen code, so a run takes no measurable time. Discovery is *not* stubbed -- it
runs the real plugin against a real test tree, which is the part most likely to break.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
RUNNER = REPO / 'etc' / 'run_mpi_tests.sh'


def _make_tree(root, *, ranks=(), marker=None):
    """A test tree declaring the given rank counts, plus one plain test."""
    root.mkdir(parents=True, exist_ok=True)
    extra = f'@pytest.mark.{marker}\n' if marker else ''
    (root / 'test_plain.py').write_text('import pytest\n\n\n' f'{extra}' 'def test_plain():\n    assert True\n')
    for n in ranks:
        (root / f'test_parallel_{n}.py').write_text(
            'import pytest\n\n\n' f'{extra}@pytest.mark.parallel({n})\n' f'def test_on_{n}():\n    assert True\n'
        )
    return root


def _stub_bin(tmp_path, exit_code=0, fail_on=None):
    """
    `mpiexec` and a `PYTEST` command that record their arguments instead of running anything.

    `fail_on` makes only the pass whose selection contains that text fail. A stub that fails every
    pass cannot detect a swallowed failure, because the last pass fails too and the script's exit
    status comes from it either way.
    """
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    log = tmp_path / 'launches.log'

    (bin_dir / 'mpiexec').write_text('#!/bin/bash\n' f'echo "mpiexec $*" >> "{log}"\n' 'shift 2\nexec "$@"\n')
    if fail_on:
        body = f'case "$*" in *"{fail_on}"*) exit 1 ;; esac\nexit 0\n'
    else:
        body = f'exit {exit_code}\n'
    (bin_dir / 'fakepytest').write_text('#!/bin/bash\n' f'echo "pytest $*" >> "{log}"\n' + body)
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    return bin_dir, log


def _run(tree, tmp_path, *, exit_code=0, marker=None, fail_on=None):
    bin_dir, log = _stub_bin(tmp_path, exit_code, fail_on)
    env = dict(os.environ)
    env['PATH'] = f"{bin_dir}{os.pathsep}{os.path.dirname(sys.executable)}{os.pathsep}{env['PATH']}"
    env['PYTEST'] = 'fakepytest'
    env['PYTHONPATH'] = f"{REPO}{os.pathsep}{env.get('PYTHONPATH', '')}"

    cmd = ['bash', str(RUNNER), str(tree)] + ([marker] if marker else [])
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=REPO)
    launches = log.read_text().splitlines() if log.exists() else []
    return proc, launches


@pytest.mark.mpi4py
def test_one_pass_per_declared_rank_count(tmp_path):
    """Each declared rank count gets its own `mpiexec` launch, and the serial pass always runs."""
    tree = _make_tree(tmp_path / 'tree', ranks=(2, 3))
    proc, launches = _run(tree, tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert 'Rank counts declared by the tests: 2 3' in proc.stdout, proc.stdout

    assert any(l.startswith('mpiexec -n 2 ') for l in launches), launches
    assert any(l.startswith('mpiexec -n 3 ') for l in launches), launches
    assert sum(l.startswith('mpiexec') for l in launches) == 2, launches

    selections = [l for l in launches if l.startswith('pytest')]
    assert any('parallel[2]' in s for s in selections), selections
    assert any('parallel[3]' in s for s in selections), selections
    assert any('parallel[1]' in s for s in selections), 'the serial pass must always run'


@pytest.mark.mpi4py
def test_tree_without_mpi_runs_one_ordinary_pass(tmp_path):
    """
    No declared ranks means one plain pass, not a `parallel[1]` selection.

    Where mpi-pytest is absent that marker is unknown and deselects everything, so the job would
    pass having run nothing at all.
    """
    tree = _make_tree(tmp_path / 'tree')
    proc, launches = _run(tree, tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert not any(l.startswith('mpiexec') for l in launches), launches
    assert len(launches) == 1, launches
    assert 'parallel[' not in launches[0], launches


@pytest.mark.mpi4py
@pytest.mark.parametrize('failing', ['parallel[2]', 'parallel[3]', 'parallel[1]'])
def test_a_failing_pass_fails_the_run(tmp_path, failing):
    """
    `shell: bash -l {0}` has no `-e`, so a failing pass must be propagated by hand.

    Each pass is failed in turn, including the first and middle ones: without `|| exit $?` the run
    would carry on and take its status from the last pass, reporting success.
    """
    tree = _make_tree(tmp_path / 'tree', ranks=(2, 3))
    proc, launches = _run(tree, tmp_path, fail_on=failing)
    assert proc.returncode != 0, f'a failure in the {failing} pass has to fail the run: {launches}'


@pytest.mark.mpi4py
def test_an_empty_selection_is_not_a_failure(tmp_path):
    """pytest exits 5 when a selection matches nothing, which is expected for some passes."""
    tree = _make_tree(tmp_path / 'tree', ranks=(2,))
    proc, _ = _run(tree, tmp_path, exit_code=5)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.mpi4py
def test_marker_is_combined_with_the_rank_selection(tmp_path):
    tree = _make_tree(tmp_path / 'tree', ranks=(2,), marker='mpi4py')
    proc, launches = _run(tree, tmp_path, marker='mpi4py')
    assert proc.returncode == 0, proc.stderr
    selections = [l for l in launches if l.startswith('pytest')]
    assert any('mpi4py and parallel[2]' in s for s in selections), selections
    assert any('mpi4py and parallel[1]' in s for s in selections), selections


@pytest.mark.base
def test_parallel_marker_without_mpi_pytest_is_an_error(monkeypatch):
    """
    A `parallel` marker in an environment without mpi-pytest must fail loudly.

    pytest ignores the unknown marker, so the test would run on one rank: passing while testing
    nothing parallel, or failing somewhere confusing. That is how a project whose environment was
    missing the plugin ran its 3-and-5-rank test on one rank and died inside qmat.

    The hook is exercised directly: a real environment without the plugin also has no entry point
    for it, which cannot be simulated by making the import fail.
    """
    import builtins
    import importlib

    sys.path.insert(0, str(REPO / 'etc'))
    mpi_ranks = importlib.import_module('mpi_ranks')

    real_import = builtins.__import__

    def no_mpi_pytest(name, *args, **kwargs):
        if name.startswith('pytest_mpi'):
            raise ImportError('mpi-pytest is not installed in this environment')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', no_mpi_pytest)

    class _Marker:
        name = 'parallel'

    class _Item:
        nodeid = 'tests/test_thing.py::test_on_two_ranks'
        own_markers = [_Marker()]

    with pytest.raises(BaseException) as excinfo:
        mpi_ranks.pytest_collection_modifyitems(None, [_Item()])
    assert 'mpi-pytest is not installed' in str(excinfo.value)

    # ...and it stays quiet when nothing declares ranks
    class _Plain:
        nodeid = 'tests/test_thing.py::test_plain'
        own_markers = []

    mpi_ranks.pytest_collection_modifyitems(None, [_Plain()])


@pytest.mark.base
def test_runner_is_posix_sh_compatible_enough_for_bash_3():
    """
    macOS still ships bash 3.2, where `"${empty[@]}"` is an unbound variable under `set -u`.

    That silently emptied rank discovery once already, so the script stays free of array expansion.
    """
    code = [l for l in RUNNER.read_text().splitlines() if not l.lstrip().startswith('#')]
    offenders = [l for l in code if '[@]' in l]
    assert not offenders, f'array expansion is unsafe under bash 3.2 with `set -u`: {offenders}'
    assert shutil.which('bash'), 'bash is needed to run the suite this way'
