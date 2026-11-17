# Only the system MPI comes from the modules: Python, NumPy, CuPy, the CUDA runtime and NCCL all come from conda-forge.
# Nothing that brings Python packages, which would shadow the environment's through PYTHONPATH: not only Python,
# SciPy-Stack or mpi4py, but also texlive, which loads the module Python. ParaStationMPI loads CUDA and NCCL modules
# as dependencies; CuPy still uses the environment's own CUDA runtime and NCCL.
module --force purge
module load Stages/2026 GCC/14.3.0 ParaStationMPI/5.13.0-1 MPI-settings/CUDA
