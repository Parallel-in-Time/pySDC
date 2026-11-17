pySDC GPU environment for JUSUF
===============================

A micromamba environment built from the same files as pySDC's GPU CI, `etc/environment-cupy.yml` and
`etc/environment-tests.yml`, so it has the packages CI tests with: Python, NumPy, CuPy, the CUDA runtime and NCCL all
come from conda-forge. Only MPI is different: conda-forge's *external* MPICH placeholder takes the place of OpenMPI,
so that mpi4py loads ParaStationMPI from the JSC modules at run time and knows the batch system and the network.

Create the environment once, on a login node (about a minute; needs `micromamba` on `PATH` or `MAMBA_EXE` set):

```bash
bash setup.sh
```

Then, in every shell or job script:

```bash
source <path to this directory>/activate.sh
```

`config.sh` sets the environment name and where micromamba keeps its files. The default is below `$SCRATCH`, which JSC
sets for the active project (`jutil env activate -p <project>`), because the environment has tens of thousands of files
and project storage is short of inodes; scratch is purged after a while, so rerun `setup.sh` when that happens.
Set `MAMBA_ROOT_PREFIX` to put it elsewhere, the same for `setup.sh` and `activate.sh`.

`modules.sh` loads only the system MPI. Any module that brings Python packages would shadow the environment's through
`PYTHONPATH`: Python, SciPy-Stack and mpi4py, but also texlive, which loads the module Python. That is why it is no
longer loaded here for LaTeX in plots.

The mpi4py-fft fork that `setup.sh` installs is needed by the GPU spectral code until pySDC owns its distributed GPU
FFT; see `docs/contrib/02_continuous_integration.md`, which also describes how to run the test suite on a Slurm
cluster.
