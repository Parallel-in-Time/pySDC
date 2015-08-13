Emulating node failure with pySDC
=================================

This example has different scripts to emulate node failures in PFASST and evaluate the outcome of different recovery 
strategies. This example folder has all the scripts to produce the figures/results from the corresponding paper, namely:

* hard_faults_test.py: generate data to check iteration counts for node failures at different steps and iterations, 
can do HEAT and ADVECTION example and all the different strategies described in the paper. 
The script postproc_hard_faults_test.py produces the corresponding heatmaps.
* hard_faults_detail.py: generate data to check residuals for a node failure at a specified step and iteration, can do
HEAT and ADVECTION example and all the different strategies described in the paper. 
The script postproc_hard_faults_detail.py produces the corresponding plots.
* grayscott_example.py: runs the 1D gray-scott example using FEniCS, emulates random node failures and dumps the residuals. 
The script postproc_grayscott.py produces the corresponding plots, the script animate_convergence.py gives some nice movies 
(very specific, not suitable for general purpose).
 
Warning: the are many global variables and hard-implemented numbers floating around in the code 
(e.g. random node failures for the gray-scott example are not allowed to occure beyond iteration 6, 
since the original example only takes 6 iterations). Also, to run the gray-scott example, FEniCS must be installed.