RDC: Rational Deferred Corrections
==================================

In this project, we integrate the rational deferred correction (RDC) method by Guettel & Klein from `this paper <http://etna.mcs.kent.edu/volumes/2011-2020/vol41/abstract.php?vol=41&pages=443-464>`_.
This is done by deriving from the equidistant collocation class, but replacing Scipy's standard BarycentricInterpolator with a custom ``MyBarycentricInterpolator``, where the blended nodes are used.
The rest is standard pySDC and all features are available.

Testing RDC convergence
-----------------------

In a first test we try to reproduce parts of Figure 4.5 (left) of the original paper.
We compute a high-resolution reference for Van der Pol's oscillator with SDC, see ``vanderpol_reference.py``.
Then, for RDC we use ``d=15`` for the blending parameter and vary the number of maximum iterations from 1 to 10.
The results can be found in ``vanderpol_error_test.py``

Multi-level RDC and PFASST with RDC
-----------------------------------

The obvious next step is to try the multi-level variant of RDC as well as a PFASST-version of RDC.
Both are not straightforward and it seems that only for smaller numbers of nodes multi-level RDC does actually converge,
at least if collocation-based coarsening is used. Yet, in ``vanderpol_MLSDC_PFASST_test.py`` we show one example where the errors as well as the mean number of iterations look fine.
Note that we use very aggressive node coarsening here, going from 20 to 2 nodes.

