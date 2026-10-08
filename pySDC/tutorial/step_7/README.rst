Step-7: pySDC with external libraries
=====================================

pySDC can be used with external libraries, in particular for spatial discretization, parallelization and solving of linear and/or nonlinear systems.
In the following, we show a few examples of pySDC + X.

- **Part A: pySDC and FEniCS.** Finite elements in space, with and without inverting the mass matrix.
- **Part B: mpi4py-fft for parallel Fourier transforms.** The nonlinear Schrödinger equation with SDC, MLSDC and PFASST.
- **Part C: Time-parallel pySDC with space-parallel PETSc.** Space and time communicators, split by coloring.
- **Part D: pySDC and PyTorch.** A tensor data type, and a network trained at the collocation nodes;
  ``torch_heat.py`` holds the data type, the heat equation using it and the network.
- **Part E: pySDC and Firedrake.** The heat equation of Part A, in Firedrake, serial, parallel across the nodes, or
  on three levels.
- **Part F: pySDC and Gusto.** pySDC as a time discretization for Gusto, in the Williamson 5 test case;
  ``F_2_plot_pySDC_with_Gusto_result.py`` plots its results.
- **Part G: pySDC on GPUs.** A heat equation in finite differences with SDC, MLSDC and PFASST, all of it on the device.

Each part is a Python script in `jupytext <https://jupytext.readthedocs.io>`_ "percent" format: run it with ``python``
(or ``mpirun``) in an environment with the library it needs, or read it on the website. Parts A, B, D, F and G also
open as notebooks in Jupyter; C and E read their process layout from the command line and only run as scripts. None of them runs in the browser or in the environment the website is built in; where the CI produces
results, the pages show them.
