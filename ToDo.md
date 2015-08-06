ToDo list
---------
   
* write a more concise requirements.md, check out [this](https://pip.pypa.io/en/latest/reference/pip_freeze.html)

* add matrix-based sweeper and e.g. compute spectral radii on the fly

* add more examples, esp. for PFASST
    - advection diffusion using fft
    - bernus model: simple explicit SDC to monodomain PFASST
    - SharpClaw integration

* write a more detailed howto, explaining all the features of the code using basic examples

* convert docs to sphinx and set up homepage

* add more detailed timings mechanism to statistics via hooks

* fix global nature of the stats object

* fix behavior of hooks during PFASST runs: initialization is called per level, but some things may need to be done only once (file openings, figures)

* add temporal coarsening