import subprocess
import os

from pySDC.tutorial.step_7.A_visualize_residuals import main as main_A
from pySDC.tutorial.step_7.B_multistep_SDC import main as main_B

def test_A():
    main_A()

def test_B():
    main_B()

def test_C_1x1():
    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    cwd = '.'
    num_procs = 1
    num_procs_space = 1
    cmd = ('mpirun -np ' + str(num_procs) + ' python pySDC/tutorial/step_7/C_pySDC_with_PETSc.py '
           + str(num_procs_space) + ' ' + 'step_7_C_out_1x1.txt').split()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env, cwd=cwd)
    p.wait()
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % \
                              (p.returncode, num_procs)

def test_C_1x2():
    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    cwd = '.'
    num_procs = 2
    num_procs_space = 2
    cmd = ('mpirun -np ' + str(num_procs) + ' python pySDC/tutorial/step_7/C_pySDC_with_PETSc.py '
           + str(num_procs_space) + ' ' + 'step_7_C_out_1x2.txt').split()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env, cwd=cwd)
    p.wait()
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % \
                              (p.returncode, num_procs)

def test_C_2x2():
    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    cwd = '.'
    num_procs = 4
    num_procs_space = 2
    cmd = ('mpirun -np ' + str(num_procs) + ' python pySDC/tutorial/step_7/C_pySDC_with_PETSc.py '
           + str(num_procs_space) + ' ' + 'step_7_C_out_2x2.txt').split()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env, cwd=cwd)
    p.wait()
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % \
                              (p.returncode, num_procs)


