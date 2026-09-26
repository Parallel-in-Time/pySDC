Step-5: PFASST
==============

In this step, we will show how pySDC can do PFASST runs (virtually parallel for now).

- **Part A: Multistep multilevel hierarchy.** How a controller holds several time steps, each with its levels.
- **Part B: My first PFASST run.** The heat equation with 1 to 16 parallel time steps.
- **Part C: Advection and PFASST.** The same for an advection problem, where the iteration counts grow.

Each part is a Python script in `jupytext <https://jupytext.readthedocs.io>`_ "percent" format: run it with
``python``, open it as a notebook in Jupyter, or read it on the website, where it is executed with every build.
