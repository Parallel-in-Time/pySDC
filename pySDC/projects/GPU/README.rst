pySDC using GPUs
================

Installation
------------
In order to start playing on GPU, install `pySDC` and its dependencies, ideally in developer mode.
First start by setting up a virtual environment, e.g. by using [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
Then also add the CuPy Package (the cuda-toolkit will be installed automatically):

    conda create -n pySDC
    conda activate pySDC
    conda install -c conda-forge --file requirements.txt
    conda install -c conda-forge cupy
When this is done (and it can take a while), you have your setup to run `pySDC` on the GPU.

Changes in the problem_classes
------------------------------
Now you have to change a little bit in the problem_classes. The first and easy step is to change the datatype.
To use pySDC on the GPU with CuPy you must use the [cupy-datatype](../../implementations/datatype_classes/cupy_mesh.py).
The next step is to import cupy in the problem_class. In the following you have to exchange the NumPy/SciPy functions with the CuPy functions.
A [Comparison Table](https://docs.cupy.dev/en/latest/reference/comparison.html) is given from CuPy to do that.
For example: The above steps can be traced using the files 
[HeatEquation_ND_FD_forced_periodic.py](../../implementations/problem_classes/HeatEquation_ND_FD_forced_periodic.py) 
and [HeatEquation_ND_FD_forced_periodic_gpu.py](../../implementations/problem_classes/HeatEquation_ND_FD_forced_periodic.py)
Now you are ready to run `pySDC` on the GPU. 

Run pySDC on the GPU
--------------------
You have to configure a script to run it. You can see at the file [heat.py](heat.py) that the parameters are the 
same for GPU and CPU. Only the import for the problem_class changed.  

More examples
-------------
Further examples can found with Allen-Cahn:
* problem: [AllenCahn_2D_FD.py](../../implementations/problem_classes/AllenCahn_2D_FD.py) and [AllenCahn_2D_FD_gpu.py](../../implementations/problem_classes/AllenCahn_2D_FD_gpu.py)
* problem: [AllenCahn_2D_FFT.py](../../implementations/problem_classes/AllenCahn_2D_FFT.py) and [AllenCahn_2D_FFT_gpu.py](../../implementations/problem_classes/AllenCahn_2D_FFT_gpu.py)
  * Script to run pySDC: [ac-fft.py](ac-fft.py)


Running large problems on GPU
-----------------------------
This project contains some infrastructure for running and plotting specific problems.
The main file is `run_experiment` and can be configured using command line arguments.
For instance, use

.. code-block:: bash
 
    srun -n 4 python work_precision.py --config=GS_USkate --procs=1/1/4 --useGPU=True --mode=run
    mpirun -np 8 python work_precision.py --config=GS_USkate --procs=1/1/4 --useGPU=True --mode=plot
    python work_precision.py --config=GS_USkate --procs=1/1/4 --useGPU=True --mode=video

to first run the problem, then make plots and then make a video for Gray-Scott with the U-Skate configuration (see arXiv:1501.01990).

To do a parallel scaling test, you can go to JUWELS Booster and use, for instance,

.. code-block:: bash
   python analysis_scripts/parallel_scaling.py --mode=run --scaling=strong --space_time=True --XPU=GPU --problem=GS
   srun python analysis_scripts/parallel_scaling.py --mode=plot --scaling=strong --space_time=True --XPU=GPU --problem=GS

This will generate jobscripts and submit the jobs. Notice that you have to wait for the jobs to complete before you can plot them.

To learn more about the options for the scripts, run them with `--help`.
