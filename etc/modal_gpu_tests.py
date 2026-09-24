"""
Run pySDC's ``cupy``-marked tests on a real GPU, using Modal's serverless containers.

These are the only tests in the project that touch a GPU, so this is the only place the
``useGPU`` code paths, the CUDA kernels behind them and NCCL are executed at all.

Run it from the repository root, either way:

    modal run etc/modal_gpu_tests.py                     # locally, needs `modal setup` once
    # or through .github/workflows/gpu_tests.yml, which passes the token in as a secret

The paths below are resolved relative to the working directory, so the repository root is the
only place this works from.
"""

import os
import pathlib

import modal

#: Where the checkout is mounted inside the container.
REMOTE = '/root/pySDC'

#: Where the coverage data lands on the machine that runs this, for the pipeline to pick up.
COVERAGE_OUT = 'coverage_GPU_hardware.dat'

#: The whole `cupy`-marked selection, which is what CI runs.
DEFAULT_TREES = ['pySDC/tests', 'pySDC/projects/GPU/tests']

#: Two of them, because `test_sweeper_NCCL` asks for two ranks and NCCL wants a GPU per rank.
#: Everything else needs one, and pays for two while it runs -- which is still a couple of cents.
GPUS = 'T4:2'

app = modal.App('pySDC-gpu-tests')

image = (
    modal.Image.micromamba(python_version='3.12')
    # The same environment a human would create to run pySDC on a GPU, so there is one place
    # describing what the GPU code needs. `mpi4py-fft` and OpenMPI are not optional here: the
    # spectral helper's `cupy`-marked tests go through them even on a single rank.
    .micromamba_install(spec_file='etc/environment-cupy.yml', channels=['conda-forge'])
    .micromamba_install(spec_file='etc/environment-tests.yml', channels=['conda-forge'])
    # Released mpi4py-fft has no `cupy`/`cupyx-scipy` FFT backend, no `DistArrayCuPy` and no
    # NCCL `comm_backend` -- upstream declined the approach in mpi4py/mpi4py-fft#14, because
    # `cupy.ndarray` cannot be subclassed the way `DistArray` needs. pySDC's GPU spectral code
    # asks for all three, so it can only run against this fork, which replaces the conda-forge
    # build installed above. Pinned to a commit rather than to the branch, so that the image is
    # reproducible and does not silently move under us.
    #
    # `c-compiler` and `cython` because the FFTW extension is built from source and its sources
    # are `.pyx`: the GitHub archive carries no pre-generated C, unlike a PyPI sdist. Without
    # Cython, setuptools quietly rewrites `utilities.pyx` to `utilities.c` and hands gcc a file
    # nothing ever generated. They are installed here rather than left to the fork's
    # `build-system.requires`, because `--no-build-isolation` is what keeps the build inside this
    # environment -- where its setup.py finds FFTW by falling back to `sys.prefix`, and where the
    # NumPy headers are the ones pySDC will run against. `--no-deps` so pip does not pull PyPI
    # wheels over the conda-forge NumPy and mpi4py.
    #
    # The source archive rather than `git+https://...`: the image has no `git`, and GitHub serves
    # the same commit as a tarball, so this pins exactly as tightly without installing one.
    #
    # `--force-reinstall` is what makes this take at all. The fork reports the same version as the
    # conda-forge build it replaces, so without it pip calls the requirement satisfied and exits
    # successfully having done nothing -- a green build and an unchanged environment. The import
    # afterwards fails the build loudly if that ever happens again: `distarrayCuPy` exists only in
    # the fork, so it is a direct check that these files, and not conda's, are installed.
    .micromamba_install('c-compiler', 'cython', channels=['conda-forge'])
    .run_commands(
        'python -m pip install --no-deps --no-build-isolation --force-reinstall '
        'https://github.com/brownbaerchen/mpi4py-fft/archive/'
        'a7aeec6ace99dd49561625c866c605ed0b337c18.tar.gz',
        'python -c "import mpi4py_fft.distarrayCuPy"',
    )
    # `mpi-pytest` supplies the `parallel` marker, which is how a test says how many ranks it
    # wants; without it `etc/run_mpi_tests.sh` finds no rank counts and runs one serial pass, and
    # the NCCL tests would quietly execute on a single rank. It is a pip package, as in the
    # `- pip:` block of every other environment file here.
    .run_commands('python -m pip install "mpi-pytest>=2026.0"')
    .env(
        {
            'PYTHONUNBUFFERED': '1',
            'PYTHONPATH': REMOTE,
            # Modal containers are root, and OpenMPI refuses to launch as root unless told twice.
            'OMPI_ALLOW_RUN_AS_ROOT': '1',
            'OMPI_ALLOW_RUN_AS_ROOT_CONFIRM': '1',
            # PMIx's shared-memory store segfaults in this container -- `PMIX ERROR: PMIX_ERROR in
            # file gds_shmem2.c` and then a dead launcher, before pytest prints anything. The hash
            # store keeps the data in process memory instead and needs nothing from the host.
            'PMIX_MCA_gds': 'hash',
            # A slot is a physical core, and this container has fewer of those than it has GPUs to
            # drive; same reasoning as the main CI pipeline's copy of these.
            'PRTE_MCA_rmaps_default_mapping_policy': ':oversubscribe',
            'OMPI_MCA_rmaps_base_oversubscribe': 'true',
        }
    )
    # `copy=False` attaches the checkout at container start instead of baking it into the image,
    # so a commit that touches only Python code reuses the cached image. The image is rebuilt
    # only when one of the two environment files changes -- which takes several minutes, because
    # conda-forge's `cupy` brings the CUDA runtime with it.
    # `.coverage*` because this does not consult `.gitignore`: a local `modal run` or pytest leaves
    # data files in the working directory, and they would be uploaded and then found by `coverage
    # combine` in the container -- a hundred of them, from other machines, on every run.
    .add_local_dir('.', REMOTE, ignore=['.git', '.claude', '**/__pycache__', '.coverage*'])
)


# `timeout` is a wall clock limit on the container, and it is the only thing that stops a hung run
# renting two GPUs until Modal's own maximum expires. A full run is about three minutes, so this is
# generous even with a cold image, and still kills a stuck one quickly. pytest's own 300 s
# per-test timeout does not help here: a deadlocked collective hangs outside any single test.
@app.function(image=image, gpu=GPUS, timeout=900)
def run_cupy_tests(trees, selection):
    """Run the GPU test suite in the container, and hand back pytest's exit code and coverage.

    Under coverage, unlike the stub runs in the main pipeline: those are excluded on purpose, so
    that lines only a GPU can reach are never reported as covered. This one really does reach
    them, so its measurement is honest and belongs in the combined report.
    """
    import subprocess

    # `etc/run_mpi_tests.sh` is how every other leg runs its tests: it asks the tests themselves
    # what rank counts they declare and launches one pass per count, so `test_sweeper_NCCL` gets
    # the two ranks its `parallel(2)` marker asks for, and everything else runs serially. Each
    # rank is wrapped so it sees a GPU of its own -- see etc/bind_gpu_to_rank.sh.
    #
    env = {
        **os.environ,
        'PYTEST': f'bash {REMOTE}/etc/bind_gpu_to_rank.sh coverage run -m pytest'
        ' --continue-on-collection-errors -v --durations=0' + (f' -k {selection}' if selection else ''),
    }
    returncode = 0
    for tree in trees:
        returncode = (
            subprocess.run(['bash', 'etc/run_mpi_tests.sh', tree, 'cupy'], cwd=REMOTE, env=env).returncode or returncode
        )

    # `concurrency = ['multiprocessing']` in pyproject.toml makes coverage write one suffixed file
    # per process, so there is no `.coverage` to read until they are combined. Not fatal if it
    # finds nothing: the test failures are the interesting output in that case, not this.
    subprocess.run(['python', '-m', 'coverage', 'combine'], cwd=REMOTE)
    measured = pathlib.Path(REMOTE, '.coverage')
    return returncode, measured.read_bytes() if measured.exists() else b''


@app.function(image=image, gpu=GPUS, timeout=900)
def run_script(path):
    """Run one script on the GPU. For profiling and one-off checks, not for CI."""
    import subprocess

    return subprocess.run(['python', path], cwd=REMOTE).returncode


@app.local_entrypoint()
def main(tests: str = ' '.join(DEFAULT_TREES), k: str = '', script: str = ''):
    """Run the GPU tests. Narrow them while developing; run the lot before pushing.

    modal run etc/modal_gpu_tests.py
    modal run etc/modal_gpu_tests.py --tests pySDC/tests/test_sweepers/test_MPI_sweeper.py
    modal run etc/modal_gpu_tests.py --k NCCL
    """
    if script:
        raise SystemExit(run_script.remote(script))

    trees = tests.split()
    returncode, measured = run_cupy_tests.remote(trees, k)

    # A narrowed run measures a fraction of the code, so its data would understate coverage rather
    # than add to it. Only a full run is worth keeping.
    if trees != DEFAULT_TREES or k:
        print('narrowed run: no coverage written')
        if returncode != 0:
            raise SystemExit(returncode)
        return

    # Coverage is measured inside the container, so it has to be carried back out as bytes; there
    # is no shared filesystem. `relative_files = true` in pyproject.toml is what makes it combine
    # with the other runners' data despite being recorded under a different absolute path.
    #
    # The name has to stay within `coverage_*.dat`, which is what the pipeline's combine steps
    # glob for, and has to differ from `coverage_GPU.dat` -- the GPU *project's* CPU test leg
    # already writes that one.
    if measured:
        pathlib.Path(COVERAGE_OUT).write_bytes(measured)
        print(f'wrote {COVERAGE_OUT} ({len(measured)} bytes)')
    else:
        print(f'no coverage data came back, so no {COVERAGE_OUT} was written')

    if returncode != 0:
        raise SystemExit(returncode)
