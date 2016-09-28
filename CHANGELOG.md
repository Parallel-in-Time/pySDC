pySDC, version 2.0
==================

Major changes
-------------

* **Complete redesign of code structure**: Now `pySDC` only contains the core 
modules and classes, while `implementations` contains the actual implementations
necessary to run something. This now includes separate files for all collocation 
classes, as well as a collection of problem and transfer classes (those used
to be in the examples, but since many examples used more or less the same 
classes, unifying this made sense). Individual implementations can still go
into the examples, of course.

* **Introduction of tutorials**: We added a few tutorials to explain many
of pySDC's features in a step-by-step fashion. We start with a simple spatial
discretization and TODO. All tutorials are accompanied by tests.

* **New all-inclusive controllers**: Instead of having two "PFASST" controllers 
which could also do SDC and MLSDC (and more), we now have four controllers
which can do all these methods, depending on the input. They are split into 
two by two class: `MPI` and `NonMPI` for real or virtual parallelisim as well
as `classic` and `multigrid` for the standard and multigrid-like implementation 
of PFASST and the likes. Initialization has been simplified a lot, too.


Minor changes
-------------

* Switched to more stable barycentric interpolation for the quadrature weights
* New collocation class: `EquidistantSpline_Right` for spline-based quadrature
* Collocation tests are realized by generators and not by classes
* Multi-step SDC (aka single-level PFASST) now works as expected


