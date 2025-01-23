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


@pytest.mark.pytorch
def test_D():
    from pySDC.tutorial.step_7.D_pySDC_with_PyTorch import train_at_collocation_nodes

    train_at_collocation_nodes()


@pytest.mark.firedrake
def test_E():
    from pySDC.tutorial.step_7.E_pySDC_with_Firedrake import runHeatFiredrake

    runHeatFiredrake(useMPIsweeper=False)


@pytest.mark.firedrake
def test_E_MPI():
    my_env = os.environ.copy()
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'
    cwd = '.'
    num_procs = 3
    cmd = f'mpiexec -np {num_procs} python pySDC/tutorial/step_7/E_pySDC_with_Firedrake.py'.split()

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env, cwd=cwd)
    p.wait()
    for line in p.stdout:
        print(line)
    for line in p.stderr:
        print(line)
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (p.returncode, num_procs)


@pytest.mark.firedrake
def test_F():
    """
    Test that the same result is obtained using the pySDC and Gusto coupling compared to only using Gusto after a few time steps.
    The test problem is Williamson 5, which involves huge numbers. Due to roundoff errors, we therefore cannot expect the solutions to match exactly.
    """
    from pySDC.tutorial.step_7.F_pySDC_with_Gusto import williamson_5
    from firedrake import norm
    import sys

    if '--running-tests' not in sys.argv:
        sys.argv += ['--running-tests']

    params = {'dt': 900, 'tmax': 2700, 'use_adaptivity': False, 'M': 2, 'kmax': 3, 'QI': 'LU'}
    stepper_pySDC, mesh = williamson_5(use_pySDC=True, **params)
    stepper_gusto, mesh = williamson_5(use_pySDC=False, mesh=mesh, **params)

    error = max(
        [
            norm(stepper_gusto.fields(comp) - stepper_pySDC.fields(comp)) / norm(stepper_gusto.fields(comp))
            for comp in ['u', 'D']
        ]
    )
    assert (
        error < 1e-8
    ), f'Unexpectedly large difference of {error} between pySDC and Gusto SDC implementations in Williamson 5 test case'
