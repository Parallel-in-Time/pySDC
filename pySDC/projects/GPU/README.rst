pySDC using GPUs
================

Installation
------------
In order to start playing on GPU, install ``pySDC`` and its dependencies, ideally in developer mode.
``etc/environment-cupy.yml`` in the repository root is the CPU dependencies plus CuPy, which brings
the CUDA toolkit with it, so creating that environment is the whole setup:

.. code-block:: bash

    conda env create -f etc/environment-cupy.yml
    conda activate pySDC

This can take a while. When it is done you are ready to run ``pySDC`` on the GPU.

.. note::

   ``environment.yml`` in *this* directory is for testing only and deliberately has no CuPy. The
   GitHub runners have no GPU, so the ``cupy``-marked tests run there against a NumPy stub (see
   ``pySDC/tests/fake_cupy.py``) and everything else runs with ``-m "not cupy"``. It will not run
   anything on a GPU -- use ``etc/environment-cupy.yml`` above. The real GPU tests run on JUWELS,
   driven by ``.gitlab-ci.yml``; the machine setups live in ``etc/venv_booster`` and
   ``etc/venv_jusuf`` next to this README.

Changes in the problem_classes
------------------------------
A problem class does not need a GPU twin. It keeps one implementation and takes a ``useGPU``
argument, and a ``setup_GPU`` classmethod swaps what the class computes with: ``xp`` from NumPy to
CuPy, ``xsp`` and ``linalg`` from SciPy's sparse modules to ``cupyx``'s, and the datatypes to
`cupy_mesh <../../implementations/datatype_classes/cupy_mesh.py>`_. The body of the class then
calls ``self.xp.sin`` where it used to call ``numpy.sin``, and works either way.

A `comparison table <https://docs.cupy.dev/en/latest/reference/comparison.html>`_ is given by CuPy
for translating the calls themselves.

`generic_ND_FD.py <../../implementations/problem_classes/generic_ND_FD.py>`_ is the example to
copy: every finite-difference problem derived from it -- the heat equation that ``heat.py`` runs,
and advection -- became GPU-capable when the base class was ported, without a line of their own.
Now you are ready to run ``pySDC`` on the GPU.

Run pySDC on the GPU
--------------------
You have to configure a script to run it. You can see at the file `heat.py <heat.py>`_ that the
parameters are the same for GPU and CPU. Only the import for the problem_class changed.

More examples
-------------
Further examples can found with Allen-Cahn. These take the other route: rather than a separate
``_gpu`` module, one class serves both and a ``useGPU`` flag switches the array, sparse and solver
modules and the datatypes over to CuPy.

* problem: `AllenCahn_2D_FD.py <../../implementations/problem_classes/AllenCahn_2D_FD.py>`_, with ``useGPU=True``
* problem: `AllenCahn_2D_FFT.py <../../implementations/problem_classes/AllenCahn_2D_FFT.py>`_, with ``useGPU=True``

  * Script to run pySDC: `ac_fft.py <ac_fft.py>`_


Running large problems on GPU
-----------------------------
This project contains some infrastructure for running and plotting specific problems.
The main file is `run_experiment` and can be configured using command line arguments.
For instance, use

.. code-block:: bash
 
    srun -n 4 python run_experiment.py --config=GS_USkate --procs=1/1/4 --useGPU=True --mode=run
    mpirun -np 8 python run_experiment.py --config=GS_USkate --procs=1/1/4 --useGPU=True --mode=plot
    python run_experiment.py --config=GS_USkate --procs=1/1/4 --useGPU=True --mode=video

to first run the problem, then make plots and then make a video for Gray-Scott with the U-Skate configuration (see arXiv:1501.01990).

To do a parallel scaling test, you can go to JUWELS Booster and use, for instance,

.. code-block:: bash

   python analysis_scripts/parallel_scaling.py --mode=run --space_time=True --XPU=GPU --problem=GS3D
   python analysis_scripts/parallel_scaling.py --mode=plot --space_time=True --XPU=GPU --problem=GS3D

This will generate jobscripts and submit the jobs. Notice that you have to wait for the jobs to complete before you can plot them.

To learn more about the options for the scripts, run them with `--help`.

Reproducing plots in Thomas Baumann's thesis
--------------------------------------------
Keep in mind that the results of the experiments are specific to the hardware that was used in the experiments.
To record the data for space-time parallel scaling experiments with Gray-Scott and RBC, run the following commands on the specified machines within the directory that contains this README.

.. code-block:: bash

    # run on JUWELS
    python analysis_scripts/parallel_scaling.py --mode=run --problem=GS3D --XPU=CPU --space_time=False
    python analysis_scripts/parallel_scaling.py --mode=run --problem=GS3D --XPU=CPU --space_time=True

    # run on JUWELS booster
    python analysis_scripts/parallel_scaling.py --mode=run --problem=GS3D --XPU=GPU --space_time=False
    python analysis_scripts/parallel_scaling.py --mode=run --problem=GS3D --XPU=GPU --space_time=True

    # run on JURECA DC
    python analysis_scripts/parallel_scaling.py --mode=run --problem=RBC --XPU=CPU --space_time=False
    python analysis_scripts/parallel_scaling.py --mode=run --problem=RBC --XPU=CPU --space_time=True

    # run on JUWELS booster
    python analysis_scripts/parallel_scaling.py --mode=run --problem=RBC --XPU=GPU --space_time=False
    python analysis_scripts/parallel_scaling.py --mode=run --problem=RBC --XPU=GPU --space_time=True

These commands will submit a bunch of jobscripts with the individual runs.
Keep in mind that these are specific to a compute project and some paths are account-specific.
Most likely, you will have to change options at the top of the file `./etc/generate_jobscript.py` before you can run anything.
Also, notice that you may not be allowed to request all resources needed for the largest Gray-Scott GPU run during normal operation of JUWELS booster.

After all jobs have run to completion, you have recorded all scaling data and may plot the results with the following command:

.. code-block:: bash

    python paper_plots.py --target=thesis

In order to run the production runs, modify the `path` class attribute of `LargeSim` in `analysis_scripts/large_simulations.py`.
Then use the following commands on the specified machines:

.. code-block:: bash

    # run on JUWELS booster
    python analysis_scripts/large_simulations.py --mode=run --problem=GS --XPU=GPU

    # run on JURECA DC
    python analysis_scripts/large_simulations.py --mode=run --problem=RBC --XPU=CPU

Plotting the results of the Gray-Scott simulation requires a lot of memory and will take very long.
Modify the paths in `analysis_scripts/plot_large_simulations.py` and then run:

.. code-block:: bash

    python analysis_scripts/3d_plot_GS_large.py --base_path=<path>
    python analysis_scripts/plot_large_simulations.py --problem=GS

Plotting the results of the Rayleigh-Benard production run is more easy.
After modifying the paths as earlier, run the following commands:

.. code-block:: bash

    python analysis_scripts/large_simulations.py --mode=plot --problem=RBC --XPU=CPU
    python analysis_scripts/large_simulations.py --mode=video --problem=RBC --XPU=CPU
    python analysis_scripts/plot_large_simulations.py --problem=RBC
    
Run scripts with `--help` to learn more about parameters.
Keep in mind that not all features are supported with all problems.
