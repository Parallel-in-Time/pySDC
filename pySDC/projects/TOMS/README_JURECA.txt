Install pySDC on JURECA
http://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JURECA/JURECA_node.html

WARNING: does not work with current Stage/2018b, need to use Stage/2018a!

> module load Intel IntelMPI
> module load petsc4py
> module load SciPy-Stack
> module load mpi4py

> pip install --user dill

> git clone https://github.com/Parallel-in-Time/pySDC.git

Go to pySDC/projects/TOMS/run_pySDC_with_PETSc.exe and change the path to pySDC in line 11.
The number behind pySDC_with_PETSc.py in line 12 gives the number of space-ranks. Take 1 nodes and 24 ntasks per node
and suitable options for this number are 1, 2, 4, 6, 12, 24, corresponding to 24, 12, 6, 4, 2, 1 ranks in time.
Currently, only 4 ntasks are requested and 1 space-rank per time-step, i.e. 4=1x4 time-steps will run in parallel.

The file pySDC/projects/TOMS/run_pySDC_with_PETSc.tmpl can be used to work with JUBE. You need to change
"param_set_spacetime" in pySDC/projects/TOMS/jube_pySDC_with_PETSc.xml accordingly.

