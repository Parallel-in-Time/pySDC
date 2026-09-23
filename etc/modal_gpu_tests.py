"""
Run pySDC's ``cupy``-marked tests on a real GPU, using Modal's serverless containers.

The rest of CI has no GPU: ``PYSDC_FAKE_GPU=1`` swaps CuPy for a NumPy stub (see
``pySDC/tests/fake_cupy.py``), which shows that the ``useGPU`` code paths are wired up and
import, but never that they compute the right thing -- and skips outright what cannot be faked.
This is the other half of that story: real CuPy on a real T4, so cuSPARSE and cuFFT results are
checked against the CPU ones, and ``test_heterogeneous_implementation`` (the only test that
skips on ``fake_cupy.ACTIVE``) runs at all.

Run it from the repository root, either way:

    modal run etc/modal_gpu_tests.py                     # locally, needs `modal setup` once
    # or through .github/workflows/gpu_tests.yml, which passes the token in as a secret

The paths below are resolved relative to the working directory, so the repository root is the
only place this works from.
"""

import pathlib

import modal

#: Where the checkout is mounted inside the container.
REMOTE = '/root/pySDC'

#: Where the coverage data lands on the machine that runs this, for the pipeline to pick up.
COVERAGE_OUT = 'coverage_GPU_hardware.dat'

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
    .env({'PYTHONUNBUFFERED': '1', 'PYTHONPATH': REMOTE})
    # `copy=False` attaches the checkout at container start instead of baking it into the image,
    # so a commit that touches only Python code reuses the cached image. The image is rebuilt
    # only when one of the two environment files changes -- which takes several minutes, because
    # conda-forge's `cupy` brings the CUDA runtime with it.
    .add_local_dir('.', REMOTE, ignore=['.git', '.claude', '**/__pycache__'])
)


@app.function(image=image, gpu='T4', timeout=1800)
def run_cupy_tests():
    """Run the GPU test suite in the container, and hand back pytest's exit code and coverage.

    Under coverage, unlike the stub runs in the main pipeline: those are excluded on purpose, so
    that lines only a GPU can reach are never reported as covered. This one really does reach
    them, so its measurement is honest and belongs in the combined report.
    """
    import subprocess

    # `PYSDC_FAKE_GPU` is deliberately *not* set: this is the run that uses the real thing.
    returncode = subprocess.run(
        [
            'python',
            '-m',
            'coverage',
            'run',
            '-m',
            'pytest',
            '-v',
            '--durations=0',
            '--continue-on-collection-errors',
            '-m',
            'cupy',
            'pySDC/tests',
            'pySDC/projects/GPU/tests',
        ],
        cwd=REMOTE,
    ).returncode

    # `concurrency = ['multiprocessing']` in pyproject.toml makes coverage write one suffixed file
    # per process, so there is no `.coverage` to read until they are combined. Not fatal if it
    # finds nothing: the test failures are the interesting output in that case, not this.
    subprocess.run(['python', '-m', 'coverage', 'combine'], cwd=REMOTE)
    measured = pathlib.Path(REMOTE, '.coverage')
    return returncode, measured.read_bytes() if measured.exists() else b''


@app.local_entrypoint()
def main():
    returncode, measured = run_cupy_tests.remote()

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
