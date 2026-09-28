Step-3: Statistics and a new sweeper
====================================

In this step, we will show how to work with the statistics pySDC generates. We will also introduce a new problem
as well as a new sweeper.

- **Part A: Getting statistics.** What the ``stats`` dictionary holds, and how to filter and sort it.
- **Part B: Adding statistics.** A hook of our own records the energy of particles in a Penning trap, solved with
  the Boris-SDC sweeper.
- **Part C: Studying collocation node types.** A parameter study over the collocation nodes, and why symmetric
  nodes conserve the energy.

Each part is a Python script in `jupytext <https://jupytext.readthedocs.io>`_ "percent" format: run it with
``python``, open it as a notebook in Jupyter, or read it on the website, where it is executed with every build.
``HookClass_Particles.py`` holds the hook of Part B, which Part C and Step 4 use as well.
