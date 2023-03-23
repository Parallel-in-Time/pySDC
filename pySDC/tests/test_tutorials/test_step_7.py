import os
import subprocess
import pytest


@pytest.mark.fenics
def test_A():
    from pySDC.tutorial.step_7.A_pySDC_with_FEniCS import main as main_A

    main_A()


@pytest.mark.mpi4py
def test_B():
    from pySDC.tutorial.step_7.B_pySDC_with_mpi4pyfft import main as main_B

    main_B()


@pytest.mark.petsc
def test_C_1x1():
    # try to import MPI here, will fail if things go wrong (and not in the subprocess part)
    try:
        import mpi4py

        del mpi4py
    except ImportError:
        raise ImportError('petsc tests need mpi4py')

    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    # my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'
    fname = 'step_7_C_out_1x1.txt'
    cwd = '.'
    num_procs = 1
    num_procs_space = 1
    cmd = (
        'mpirun -np '
        + str(num_procs)
        + ' python pySDC/tutorial/step_7/C_pySDC_with_PETSc.py '
        + str(num_procs_space)
        + ' '
        + fname
    ).split()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env, cwd=cwd)
    p.wait()
    for line in p.stdout:
        print(line)
    for line in p.stderr:
        print(line)
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (p.returncode, num_procs)


@pytest.mark.petsc
def test_C_1x2():
    # try to import MPI here, will fail if things go wrong (and not in the subprocess part)
    try:
        import mpi4py
    except ImportError:
        raise ImportError('petsc tests need mpi4py')
    finally:
        del mpi4py

    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'
    cwd = '.'
    fname = 'step_7_C_out_1x2.txt'
    num_procs = 2
    num_procs_space = 2
    cmd = (
        'mpirun -np '
        + str(num_procs)
        + ' python pySDC/tutorial/step_7/C_pySDC_with_PETSc.py '
        + str(num_procs_space)
        + ' '
        + fname
    ).split()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env, cwd=cwd)
    p.wait()
    for line in p.stdout:
        print(line)
    for line in p.stderr:
        print(line)
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (p.returncode, num_procs)


@pytest.mark.petsc
def test_C_2x2():
    # try to import MPI here, will fail if things go wrong (and not in the subprocess part)
    try:
        import mpi4py

        del mpi4py
    except ImportError:
        raise ImportError('petsc tests need mpi4py')

    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'
    cwd = '.'
    fname = 'step_7_C_out_2x2.txt'
    num_procs = 4
    num_procs_space = 2
    cmd = (
        'mpirun -np '
        + str(num_procs)
        + ' python pySDC/tutorial/step_7/C_pySDC_with_PETSc.py '
        + str(num_procs_space)
        + ' '
        + fname
    ).split()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env, cwd=cwd)
    p.wait()
    for line in p.stdout:
        print(line)
    for line in p.stderr:
        print(line)
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (p.returncode, num_procs)
