import pytest
import subprocess

from pySDC.projects.AllenCahn_Bayreuth.run_simple_forcing_verification import main, visualize_radii

def test_main_serial():
    main()
    visualize_radii()

@pytest.mark.slow
@pytest.mark.parallel
def test_main_parallel():

    # try to import MPI here, will fail if things go wrong (and not later on in the subprocess part)
    import mpi4py

    nprocs = 2
    cmd = f"export PYTHONPATH=$PYTHONPATH:$(pwd); mpirun -np {nprocs} python pySDC/projects/AllenCahn_Bayreuth/run_simple_forcing_benchmark.py -n {nprocs}"
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
    p.wait()
    (output, err) = p.communicate()
    print(output)
    assert err == ''

    nprocs = 4
    cmd = f"export PYTHONPATH=$PYTHONPATH:$(pwd); mpirun -np {nprocs} python pySDC/projects/AllenCahn_Bayreuth/run_simple_forcing_benchmark.py -n {nprocs}"
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
    p.wait()
    (output, err) = p.communicate()
    print(output)
    assert err == ''
