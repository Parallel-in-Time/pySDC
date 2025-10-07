import pytest


def get_composite_collocation_problem(L, M, N, alpha=0):
    import numpy as np
    from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
    from pySDC.implementations.controller_classes.controller_ParaDiag_nonMPI import controller_ParaDiag_nonMPI

    from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d as problem_class
    from pySDC.implementations.sweeper_classes.ParaDiagSweepers import QDiagonalization as sweeper_class

    problem_params = {'lambdas': -1.0 * np.ones(shape=(N)), 'u0': 1}

    level_params = {}
    level_params['dt'] = 1e-1
    level_params['restol'] = -1

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = M
    sweeper_params['initial_guess'] = 'copy'
    sweeper_params['update_f_evals'] = True

    step_params = {}
    step_params['maxiter'] = 1

    controller_params = {}
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = []
    controller_params['mssdc_jac'] = False
    controller_params['alpha'] = alpha

    description = {}
    description['problem_class'] = problem_class
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper_class
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    controller_args = {
        'controller_params': controller_params,
        'description': description,
    }
    controller = controller_ParaDiag_nonMPI(**controller_args, num_procs=L)
    P = controller.MS[0].levels[0].prob

    for prob in [S.levels[0].prob for S in controller.MS]:
        prob.init = tuple([*prob.init[:2]] + [np.dtype('complex128')])

    return controller, P


@pytest.mark.base
@pytest.mark.parametrize('M', [1, 3])
@pytest.mark.parametrize('N', [2, 4])
@pytest.mark.parametrize('ignore_ic', [True, False])
def test_direct_solve(M, N, ignore_ic):
    """
    Test that the diagonalization has the same result as a direct solve of the collocation problem
    """
    import numpy as np
    import scipy.sparse as sp

    controller, prob = get_composite_collocation_problem(1, M, N)

    controller.MS[0].levels[0].status.unlocked = True
    level = controller.MS[0].levels[0]
    level.status.time = 0
    sweep = level.sweep
    sweep.params.ignore_ic = ignore_ic

    # initial conditions
    for m in range(M + 1):
        level.u[m] = prob.u_exact(0)
        level.f[m] = prob.eval_f(level.u[m], 0)

    level.sweep.compute_residual()

    if ignore_ic:
        level.u[0][:] = None

    sweep.update_nodes()
    sweep.eval_f_at_all_nodes()

    # solve directly
    I_MN = sp.eye((M) * N)
    Q = sweep.coll.Qmat[1:, 1:]
    C_coll = I_MN - level.dt * sp.kron(Q, prob.A)

    u0 = np.zeros(shape=(M, N), dtype=complex)
    for m in range(M):
        u0[m, ...] = prob.u_exact(0)
    u = sp.linalg.spsolve(C_coll, u0.flatten()).reshape(u0.shape)

    for m in range(M):
        if ignore_ic:
            level.u[m + 1] = level.u[m + 1] + level.increment[m]
        assert np.allclose(u[m], level.u[m + 1])

    if not ignore_ic:
        sweep.compute_residual()
        assert np.isclose(level.status.residual, 0), 'residual is non-zero'


if __name__ == '__main__':
    test_direct_solve(2, 1, True)
