Step-6: Advanced PFASST controllers
===================================

We discuss controller implementations, features and parallelization of PFASST controllers in this step.

- **Part A: The non-MPI controller.** SDC, and MLSDC and PFASST on 1, 2, 4 and 8 processes, emulated in one process.
- **Part B: Odd temporal distribution.** 3, 5, 7 and 9 processes for 8 time steps.
- **Part C: MPI parallelization.** The same with ``controller_MPI``, one step per MPI rank, and the check that it
  gives the same results.

Each part is a Python script in `jupytext <https://jupytext.readthedocs.io>`_ "percent" format: run it with
``python`` (Part C with ``mpirun``), open it as a notebook in Jupyter, or read it on the website, where Parts A and B
are executed with every build. ``pfasst_setup.py`` holds the setup all three share.
