Resilience in Block Gauss-Seidel SDC
====================================
Here, we explore how to deal with soft faults in both serial SDC as well as parallel-in-time block Gauss-Seidel mulit-step SDC.
We have two strategies for now: Adaptivity, which is originally a scheme for providing dynamic refinement in time, but thanks to restarts in the refinement algorithm we get some resilience here, and Hot Rod, which is an explicit detector for soft faults.
 
.. include:: Adaptivity.rst
.. include:: HotRod.rst
