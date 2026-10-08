Step-8: Advanced topics
=======================

Since we now have explored the basic features of pySDC, we are actually ready to do some serious science (e.g. in the
projects). However, we gather here further interesting cases, e.g. special flags or more alternative implementations
of components.

- **Part A: Adaptive time-stepping.** Step sizes from an embedded error estimate, with restarts, against fixed steps.
- **Part B: Multi-step SDC.** Parallel time steps on a single level, in a parallel and a serial variant.
- **Part C: Iteration estimator.** Stopping when the estimated error, not the residual, is small enough.

Each part is a Python script in `jupytext <https://jupytext.readthedocs.io>`_ "percent" format: run it with
``python``, open it as a notebook in Jupyter, or read it on the website, where it is executed with every build.
``HookClass_error_output.py`` holds the hook of Part C.
