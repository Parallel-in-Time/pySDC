ToDo list
---------

* simplify stats creation and collecting:
so far, the stats are defined per step, per level and per iteration (which is good, I think), but in order to get data 
into these structures, one needs to (1) explicitly attach a new iteration stats object to a level stats object (which is
ok, but should be done automatically somehow) and (2) explicitly write things like the residual into the structure 
(which is really bad and should at least be encapsulated). 
    
* write a more concise requirements.md, check out [this](https://pip.pypa.io/en/latest/reference/pip_freeze.html)

* add matrix-based sweeper and e.g. compute spectral radii on the fly

* add real parallelization for PFASST using mpi4py

* add more examples, esp. for PFASST
    - advection diffusion using fft
    - reaction diffusion using fft
    - bernus model: simple explicit SDC to monodomain PFASST
    - multi-particle penning trap with MLSDC and PFASST
    - SWFW, maybe couple to PyClaw

* write a more detailed howto, explaining all the features of the code using basic examples

* convert docs to sphinx and set up homepage