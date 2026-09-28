Step-9: ParaDiag
================

ParaDiag is a parallel-in-time method of a rather different flavour than PFASST.
Instead of iterating on a hierarchy of levels and passing information forward step by step, it diagonalizes the "top layer" of Kronecker products that makes up the composite collocation problem.
After the diagonalization, the collocation problems on the individual steps decouple and can be solved concurrently, which is where the parallelism comes from.

The price is an approximation: the time-stepping matrix is replaced by an :math:`\alpha`-circulant one, which *is* diagonalizable by a weighted Fourier transform.
The outer iteration then corrects for that perturbation, and :math:`\alpha` trades approximation quality against the conditioning of the diagonalization.

- **Part A: ParaDiag for linear problems.** The method as a few lines of linear algebra.
- **Part B: ParaDiag for nonlinear problems.** Averaging the Jacobian across the steps.
- **Part C: ParaDiag in pySDC.** Through pySDC's controllers, compared with PFASST and serial time stepping.
- **Part D: Adaptive alpha.** Letting ParaDiag choose its alpha by itself.
- **Part E: MPI-parallel ParaDiag.** The same, with one time step per MPI rank.

Each part is a Python script in `jupytext <https://jupytext.readthedocs.io>`_ "percent" format: run it with
``python`` (Part E with ``mpirun``), open it as a notebook in Jupyter, or read it on the website, where Parts A to D
are executed with every build. ``paradiag_setup.py`` holds the setup Parts D and E share.
