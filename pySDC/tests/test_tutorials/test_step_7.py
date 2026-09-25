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
@pytest.mark.parallel(1)
def test_C_1x1():
    from pySDC.tutorial.step_7.C_pySDC_with_PETSc import main as main_C

    main_C(num_procs_space=1, fname='step_7_C_out_1x1.txt')


@pytest.mark.petsc
@pytest.mark.parallel(2)
def test_C_1x2():
    from pySDC.tutorial.step_7.C_pySDC_with_PETSc import main as main_C

    main_C(num_procs_space=2, fname='step_7_C_out_1x2.txt')


@pytest.mark.petsc
@pytest.mark.parallel(4)
def test_C_2x2():
    from pySDC.tutorial.step_7.C_pySDC_with_PETSc import main as main_C

    main_C(num_procs_space=2, fname='step_7_C_out_2x2.txt')


@pytest.mark.pytorch
def test_D():
    import numpy as np
    import torch

    torch.manual_seed(42)
    np.random.seed(42)

    from pySDC.tutorial.step_7.D_pySDC_with_PyTorch import train_at_collocation_nodes

    train_at_collocation_nodes()


@pytest.mark.firedrake
@pytest.mark.parametrize('ML', [True, False])
def test_E(ML):
    from pySDC.tutorial.step_7.E_pySDC_with_Firedrake import runHeatFiredrake

    runHeatFiredrake(useMPIsweeper=False, ML=ML)


@pytest.mark.firedrake
@pytest.mark.parallel(3)
def test_E_MPI():
    from pySDC.tutorial.step_7.E_pySDC_with_Firedrake import runHeatFiredrake

    runHeatFiredrake(useMPIsweeper=True)


@pytest.mark.firedrake
def test_F(monkeypatch):
    """
    Test that the same result is obtained using the pySDC and Gusto coupling compared to only using Gusto after a few time steps.
    The test problem is Williamson 5, which involves huge numbers. Due to roundoff errors, we therefore cannot expect the solutions to match exactly.
    """
    from pySDC.tutorial.step_7.F_pySDC_with_Gusto import williamson_5
    from firedrake import norm
    import sys

    monkeypatch.setattr(sys, 'argv', [*sys.argv, '--running-tests'])

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


@pytest.mark.firedrake
def test_F_ML(monkeypatch):
    """
    Test that the Gusto coupling with multiple levels in space converges
    """
    from pySDC.tutorial.step_7.F_pySDC_with_Gusto import williamson_5
    from pySDC.helpers.stats_helper import get_sorted, filter_stats
    import sys

    monkeypatch.setattr(sys, 'argv', [*sys.argv, '--running-tests'])

    params = {'use_pySDC': True, 'dt': 1000, 'tmax': 1000, 'use_adaptivity': False, 'M': 2, 'kmax': 4, 'QI': 'LU'}
    stepper_ML, _ = williamson_5(Nlevels=2, **params)
    stepper_SL, _ = williamson_5(Nlevels=1, **params)

    stats = stepper_ML.scheme.stats
    residual_fine = get_sorted(stats, type='residual_post_sweep', level=0, sortby='iter')
    residual_coarse = get_sorted(stats, type='residual_post_sweep', level=1, sortby='iter')
    assert residual_fine[0][1] / residual_fine[-1][1] > 1e2, 'Fine residual did not converge as expected'
    assert residual_coarse[0][1] / residual_coarse[-1][1] > 3e2, 'Coarse residual did not converge as expected'

    stats_SL = stepper_SL.scheme.stats
    residual_SL = get_sorted(stats_SL, type='residual_post_sweep', sortby='iter')
    assert all(
        res_SL > res_ML for res_SL, res_ML in zip(residual_SL, residual_fine, strict=True)
    ), 'Single level SDC converged faster than multi-level!'


@pytest.mark.cupy
@pytest.mark.parallel(2)
def test_G():
    from pySDC.tutorial.step_7.G_pySDC_on_GPU import main as main_G

    main_G()
