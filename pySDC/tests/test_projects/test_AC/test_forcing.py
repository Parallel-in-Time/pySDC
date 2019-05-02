import subprocess
import os

import numpy as np
from mpi4py import MPI

from pySDC.projects.AllenCahn_Bayreuth.run_simple_forcing_verification import main, visualize_radii

# def test_main_serial():
#     main()
#     visualize_radii()

def test_main_parallel():

    # try to import MPI here, will fail if things go wrong (and not later on in the subprocess part)
    import mpi4py

    # my_env = os.environ.copy()
    # my_env['PYTHONPATH'] = '../../..:.'
    nprocs = 2
    cmd = f"mpirun -np {nprocs} python pySDC/projects/AllenCahn_Bayreuth/run_simple_forcing_problems.py -n {nprocs}"
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
    p.wait()
    (output, err) = p.communicate()
    print(output)

    nprocs = 4
    cmd = f"mpirun -np {nprocs} python pySDC/projects/AllenCahn_Bayreuth/run_simple_forcing_problems.py -n {nprocs}"
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
    p.wait()
    (output, err) = p.communicate()
    print(output)

