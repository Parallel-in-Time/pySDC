Step-4: Multilevel SDC
======================

In this step, we will show how pySDC creates a multilevel hierarchy and how MLSDC can be run and tested.

- **Part A: Spatial transfer operators.** Interpolation between two resolutions, and its order.
- **Part B: Multilevel hierarchy.** How lists in the description become levels.
- **Part C: SDC vs. MLSDC.** Half the iterations, but not half the work.
- **Part D: MLSDC with particles.** Coarsening by a simpler problem, for the Penning trap of Step 3.

Each part is a Python script in `jupytext <https://jupytext.readthedocs.io>`_ "percent" format: run it with
``python``, open it as a notebook in Jupyter, or read it on the website, where it is executed with every build.
``PenningTrap_3D_coarse.py`` holds the coarse problem of Part D.
