pySDC
======

The pySDC project is a Python implementation of the spectral deferred correction (SDC) approach and its flavors, 
esp. the multilevel extension MLSDC. It is intended for rapid prototyping and educational purposes. New ideas like e.g 
sweepers or predictors can be tested and first toy problems can be easily implemented. Two virtually parallel PFASST 
iterators are implemented as well, giving full access to all values at any time.


News
----

* June 17, 2015: Gray-Scott example using FEniCS is up and running
* May 7, 2015: added first [FEniCS](http://fenicsproject.org/) example
* May 5, 2015: added [Clawpack](http://www.clawpack.org/) examples using Sharpclaw
* April 16, 2015: new PFASST iterator (loop-based) implemented, new examples added
* March 19, 2015: code now also runs with Python 2.7, development of SharpClaw example started 
* January 7, 2015: revised the examples to work with the new driver in Methods.py, new statistics framework
* December 29, 2014: virtual PFASST implemented using stages, see [flowchart](flowchart.png) for implementation details
* November 4, 2014: First open source release on github, four very basic examples up and running, code is documented


Documentation and Testing
-------------------------

Most of the source code is documented (besides some of the examples, at least for now). 
For doxygen generated documentation a Doxyfile is provided. The latest, auto-generated documentation can be found on 
the [PinT server](https://pint.fz-juelich.de/ci/view/pySDC/job/PYSDC_DOCU/doxygen). To compile your 
own documentation, use the [doxypypy](https://github.com/Feneric/doxypypy) filter. 

Some first, rather rudimentary tests can be found in the tests directory and nose should be able to run those 
out-of-the-box. Auto-generated test results are here: 
[![status-img][]](https://travis-ci.org/Parallel-in-Time/pySDC)


HowTo
-----

To start your own example, take a look at soem of the examples shipped with this code:

* heat1d: MLSDC and PFASST implementation of the forced 1D heat equation with Dirichlet-0 BC in [0,1]
* penningtrap: particles in a penning trap, driven by external electric and magnetic fields
* vanderpol: the van der pol oscillator

To run one of these, add the root directory of pySDC to your PYTHONPATH and execute `python playground` (this could 
be done e.g. via `PYTHONPATH=../.. python playground.py`). 

Each of these examples should demonstrate some features of this code, e.g. MLSDC/PFASST and an IMEX sweeper for the heat 
equation, the Boris-SDC approach in the particle case and the LU decomposition as well as the application of a 
nonlinear solver in the van der pol example.
 
For a new example, you have to either choose or provide at least five components:

* the collocation, examples can be found in pySDC/CollocationClasses.py
* a problem description, examples can be found in examples/*/ProblemClass.py
* a data type, examples can be found in pySDC/datatype_classes/
* a sweeper, examples can be found in pySDC/sweeper_classes/
* a method/stepper, provided in the drivers pySDC/PFASST_*.py 


For MLSDC, suitable transfer operators are also required, examples can be found e.g. in examples/heat1d/TransferClass.py.

The playground.py routines in the examples show how these components need to be linked together. Basically, 
most of the management is done via the level and the step data structures. Here, 
the components are coupled as expected by the method and all the other components.

In the easiest case (where collocation, data type, sweeper, method, hooks and transfer operators can be used as 
provided with this code), only a custom problem description have to be implemented.

Note: all interfaces are subject to changes, if necessary.


[status-img]: https://travis-ci.org/Parallel-in-Time/pySDC.svg?branch=master