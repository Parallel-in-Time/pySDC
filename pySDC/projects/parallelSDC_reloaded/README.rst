Numerical experiment scripts for the parallel SDC paper
==============================================================

Python scripts of the numerical experiment for the following paper:

.. code-block:: tex

    @misc{caklovicXXXXimproving,
        title={Improving Parallelism Across the Method for Spectral Deferred Corrections},
        author={Gayatri \v{C}aklovi\'c and Lunet Thibaut and Götschel Sebastian and Ruprecht Daniel},
        year={2023},
        comment={to be submitted to SISC},
    }

Figures for the manuscript
--------------------------

See the `scripts` folder, all python scripts have the `fig0[...]` prefix, and each script generate several figures 
from the manuscript.
One can run everything with the `run.sh` script, and crop the pdf figures to the format used in the manuscript with the `crop.sh` script.

Note that most scripts generate error vs cost figure using a cost modeling of the simulation that counts the
number of Newton iterations and RHS evaluation.
This can be switched to a estimation of the cost using runtime measurements by :

- modifying the return statement in the `getCost` function by `return tComp`
- changing the `xlabel="Cost"` in the `plt.gca().set` call into `xlabel="Computation time [s]"`
- commenting the `xlim` argument in the `plt.gca().set` call

Experimental scripts
--------------------

For several problem `probName`, there is two scripts :

- `{probName}_setup.py` : runs the problem with specific parameters and one given SDC configuration
- `{probName}_accuracy.py` : generate error vs dt and error vs cost figures for different SDC configurations

In addition, there is those generic scripts for analysis :

- `convergence.py` : generate convergence graph for specific SDC configurations, using Dahlquist
- `nilpotency.py` : look at nilpotency of stiff and non-stiff limit of the SDC iteration matrix
- `stability.py` : generate stability contours for specific SDC configurations

Finally, all those scripts use the utility module `utils.py`.
