import pytest
import numpy as np


@pytest.mark.base
def test_predict_main():
    from pySDC.projects.DAE.problems.simple_DAE import simple_dae_1
    from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
    from pySDC.core.step import Step

    # initialize level parameters
    level_params = dict()
    level_params['dt'] = 5e-2

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['initial_guess'] = 'zero'

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-12  # tollerance for implicit solver

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = simple_dae_1
    description['problem_params'] = problem_params
    description['sweeper_class'] = fully_implicit_DAE
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params

    S = Step(description=description)
    L = S.levels[0]
    P = L.prob
    # set initial time in the status of the level
    L.status.time = 0.1
    # compute initial value (using the exact function here)
    L.u[0] = P.u_exact(L.time)
    # call prediction function to initialise nodes
    L.sweep.predict()
    # check correct initialisation
    assert np.allclose(abs(L.f[0]), 0.0)
    for i in range(sweeper_params['num_nodes']):
        assert np.allclose(abs(L.u[i + 1]), 0.0)
        assert np.allclose(abs(L.f[i + 1]), 0.0)

    # rerun check for random initialisation
    # expecting that random initialisation does not initialise to zero
    sweeper_params['initial_guess'] = 'random'
    description['sweeper_params'] = sweeper_params
    S = Step(description=description)
    L = S.levels[0]
    P = L.prob
    # set initial time in the status of the level
    L.status.time = 0.1
    # compute initial value (using the exact function here)
    L.u[0] = P.u_exact(L.time)
    L.sweep.predict()
    assert abs(L.f[0]) == 0.0
    for i in range(sweeper_params['num_nodes']):
        assert abs(L.u[i + 1]) > 0.0
        assert abs(L.f[i + 1]) > 0.0


@pytest.mark.base
def test_residual_main():
    from pySDC.projects.DAE.problems.simple_DAE import simple_dae_1
    from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
    from pySDC.core.step import Step

    # initialize level parameters
    level_params = dict()
    level_params['dt'] = 1e-1
    level_params['residual_type'] = 'last_abs'

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-12  # tollerance for implicit solver

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = simple_dae_1
    description['problem_params'] = problem_params
    description['sweeper_class'] = fully_implicit_DAE
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    # description['step_params'] = step_params

    # last_abs residual test
    S = Step(description=description)
    L = S.levels[0]
    P = L.prob
    # set reference values
    u = P.dtype_u(P.init)
    du = P.dtype_u(P.init)
    u.diff[:2] = (5, 5)
    u.alg[0] = 5
    du.diff[:2] = (0, 0)
    du.alg[0] = 0
    # set initial time in the status of the level
    L.status.time = 0.0
    L.u[0] = u
    # call prediction function to initialise nodes
    L.sweep.predict()
    L.sweep.compute_residual()
    # generate reference norm
    ref_norm = []
    for m in range(3):
        ref_norm.append(abs(P.eval_f(u, du, L.time + L.dt * L.sweep.coll.nodes[m])))
    # check correct residual computation
    assert L.status.residual == ref_norm[-1], "ERROR: incorrect norm used"

    # full_rel residual test
    level_params['residual_type'] = 'full_rel'
    description['level_params'] = level_params

    S = Step(description=description)
    L = S.levels[0]
    P = L.prob
    # set initial time in the status of the level
    L.status.time = 0.0
    # compute initial value (using the exact function here)
    L.u[0] = u
    # call prediction function to initialise nodes
    L.sweep.predict()
    L.sweep.compute_residual()
    assert L.status.residual == max(ref_norm) / abs(L.u[0]), "ERROR: incorrect norm used"

    # last_rel residual test
    level_params['residual_type'] = 'last_rel'
    description['level_params'] = level_params

    S = Step(description=description)
    L = S.levels[0]
    P = L.prob
    # set initial time in the status of the level
    L.status.time = 0.0
    # compute initial value (using the exact function here)
    L.u[0] = u
    # call prediction function to initialise nodes
    L.sweep.predict()
    L.sweep.compute_residual()
    assert L.status.residual == ref_norm[-1] / abs(L.u[0]), "ERROR: incorrect norm used"


@pytest.mark.base
def test_compute_end_point_main():
    from pySDC.projects.DAE.problems.simple_DAE import simple_dae_1
    from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
    from pySDC.core.step import Step

    # initialize level parameters
    level_params = dict()
    level_params['dt'] = 1e-1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['initial_guess'] = 'zero'

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-12  # tollerance for implicit solver

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = simple_dae_1
    description['problem_params'] = problem_params
    description['sweeper_class'] = fully_implicit_DAE
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params

    # last_abs residual test
    S = Step(description=description)
    L = S.levels[0]
    P = L.prob
    # set initial time in the status of the level
    L.status.time = 0.0
    # compute initial value (using the exact function here)
    L.u[0] = P.u_exact(L.time)
    # call prediction function to initialise nodes
    L.sweep.predict()
    # computer end point
    L.sweep.compute_end_point()

    for m in range(1, L.sweep.coll.num_nodes):
        assert np.allclose(abs(L.u[m] - L.uend), 0.0), "ERROR: end point not computed correctly"
