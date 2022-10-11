pySDC using GPUs
===================
Installation
------------
In order to start playing on GPU, install `pySDC` and it's dependencies, ideally in developer mode.
First start by setting up a virtual environment, e.g. by using [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
Then also add the CuPy Package (the cuda-toolkit will be installed automatically):

    conda create -n pySDC
    conda activate pySDC
    conda install -c conda-forge --file requirements.txt
    conda install -c conda-forge cupy
When this is done (and it can take a while), you have your setup to run `pySDC` on the GPU.

Changes in the problem_classes
------------
Now you have to change a little bit in the problem_classes. The first and easy step is to change the datatype.
To use pySDC on the GPU with CuPy you must use the [cupy-datatype](../../implementations/datatype_classes/cupy_mesh.py).
The next step is to import cupy in the problem_class. In the following you have to exchange the NumPy/SciPy functions with the CuPy functions.
A [Comparison Table](https://docs.cupy.dev/en/latest/reference/comparison.html) is given from CuPy to do that.
For Exmaple: The above steps can be traced using the files 
[HeatEquation_ND_FD_forced_periodic.py](../../implementations/problem_classes/HeatEquation_ND_FD_forced_periodic.py) 
and [HeatEquation_ND_FD_forced_periodic_gpu.py](../../implementations/problem_classes/HeatEquation_ND_FD_forced_periodic.py)
Now you are ready to run `pySDC` on the GPU. 

Run pySDC on the GPU
------------
You have to configure a Script to run it. You can see at the file [heat.py](heat.py) that the parameters are the 
same for GPU and CPU. Only the import for the problem_class changed.  



More examples
----------
Further examples can found with Allen-Cahn:
* problem: [AllenCahn_2D_FD.py](../../implementations/problem_classes/AllenCahn_2D_FD.py) and [AllenCahn_2D_FD_gpu.py](../../implementations/problem_classes/AllenCahn_2D_FD_gpu.py)
* problem: [AllenCahn_2D_FFT.py](../../implementations/problem_classes/AllenCahn_2D_FFT.py) and [AllenCahn_2D_FFT_gpu.py](../../implementations/problem_classes/AllenCahn_2D_FFT_gpu.py)
  * Script to run pySDC: [ac-fft.py](ac-fft.py)


