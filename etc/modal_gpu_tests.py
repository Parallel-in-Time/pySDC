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

import modal

#: Where the checkout is mounted inside the container.
REMOTE = '/root/pySDC'

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
    # `c-compiler` because the FFTW extension is built from source; it finds FFTW on its own,
    # since the fork's setup.py falls back to `sys.prefix` and conda-forge put it there. No
    # build isolation, so the build sees the environment's own NumPy and setuptools, and
    # `--no-deps` so pip does not pull PyPI wheels over the conda-forge NumPy and mpi4py.
    .micromamba_install('c-compiler', channels=['conda-forge'])
    .run_commands(
        'python -m pip install --no-deps --no-build-isolation '
        'git+https://github.com/brownbaerchen/mpi4py-fft.git@a7aeec6ace99dd49561625c866c605ed0b337c18'
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
    """Run the GPU test suite in the container and hand back pytest's exit code."""
    import subprocess

    # `PYSDC_FAKE_GPU` is deliberately *not* set: this is the run that uses the real thing.
    return subprocess.run(
        [
            'python',
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


@app.local_entrypoint()
def main():
    returncode = run_cupy_tests.remote()
    if returncode != 0:
        raise SystemExit(returncode)
