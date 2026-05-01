SDC for Rayleigh-Benard convection
==================================

Run simulations
---------------
In order to run simulations, use commands like

.. code-block:: bash
   cd pySDC/projects/RayleighBenard
   mpirun -np 64 python run_experiment.py --res=32 --dt=0.06 --config=RBC3DG4R4SDC23Ra1e5 --procs=1/2/32 --mode=run --useGPU=False

use

.. code-block:: bash
   python run_experiment.py --help

to get more information about the different parameters. The config names always start with ``RBC3DG4R4``, which means Rayleigh-Benard convection in 3D with aspect ratio four and four times as many degrees of freedom in horizontal directions as in the vertical.
Next comes the time-stepping scheme. Choices are ``SDC44``, ``SDC23``, ``RK`` for RK443 and ``Euler`` for RK111. Finally, the Rayleigh number is specified using ``Ra1e5``, ``Ra1e6``, or ``Ra1e7``.
Note that you need to run the simulations in order of ascending Rayleigh number since larger Rayleigh number simulations take solutions from lower Rayleigh number experiments as initial conditions. Only ``Ra=1e5`` is started from random perturbations.

To analyse, stay in the directory you ran the simulation in and use commands like

.. code-block:: bash
   python analysis_scripts/process_RBC3D_data.py --config=RBC3DG4R4SDC23Ra1e5 --dt=0.06 --res=32


Benchmarks
----------
The benchmarks use JUBE.
Please run them using commands like

.. code-block:: bash
   module load JUBE
   cd pySDC/projects/RayleighBenard/benchmarks
   OUT=JUSUF_RBC3DG4R4SDC44Ra1e5 jube run jube_script.yaml -t JUSUF SDC44 Ra1e5
   jube result bench_run_JUSUF_RBC3DG4R4SDC44Ra1e5 -a > results/JUSUF_RBC3DG4R4SDC44Ra1e5.txt
Use tags ``JUSUF`` of ``BOOSTER`` for running on JUSUF or JUWELS booster respectively. The tags for configurations are ``RBC3DG4R4SDC44Ra1e5`` and ``RBC3DG4R4SDC44Ra1e6``

Once you have run all the benchmarks, plot them with

.. code-block:: bash
    cd pySDC/projects/RayleighBenard
    python analysis_scripts/plot_benchmarks.py


Plotting the order of accuracy
------------------------------
For this you first need to compute the error for all configurations (``RBC3DG4R4SDC44Ra1e5``, ``RBC3DG4R4SDC23Ra1e5``, ``RBC3DG4R4RKRa1e5``, and ``RBC3DG4R4EulerRa1e5``) in the plot.
However, before you can run these simulations, make sure, you have run the configuration ``RBC3DG4R4SDC23Ra1e5`` to get initial conditions.
Once you have those available, use

.. code-block:: bash
    cd pySDC/projects/RayleighBenard
    mpirun -np 64 python analysis_scripts/RBC3D_order.py --config=RBC3DG4R4SDC23Ra1e5 --procs=1/1/64 --useGPU=False --mode=run

for all the configurations to generate the data and then run

.. code-block:: bash
    python analysis_scripts/RBC3D_order.py

to make the plot.


Plotting microscopic verification
---------------------------------
After you have plotted the order of accuracy, you can make a plot for microscopic verification, which includes the order of accuracy plot and adds a plot for the spectrum.
You need to run and analyse simulations with:
 - ``--res=32 --dt=0.06 --config=RBC3DG4R4SDC23Ra1e5``
 - ``--res=64 --dt=0.01 --config=RBC3DG4R4SDC23Ra1e6``
 - ``--res=128 --dt=0.005 --config=RBC3DG4R4SDC23Ra1e7``

Then, just run

.. code-block:: bash
    python analysis_scripts/RBC3D_spectrum.py

to make the plot.


Plotting macroscopic verification
---------------------------------
Macroscopic verification is done via comparison with data from https://doi.org/10.5281/zenodo.14205874.
You need to download this reference dataset and copy it to `pySDC/projects/RayleighBenard/data/Nek5000`.
Apart from that you need the pySDC simulation data prepared in the microscopic verification step.
Then you simply run

.. code-block:: bash
    python analysis_scripts/compare_Nek5000.py

to make the plot.
