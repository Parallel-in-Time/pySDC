import pytest


def getRandomArrays(case):
    """
    Returns random arrays and values for test.

    Parameters
    ----------
    case : int
        Index of test case.

    Returns
    -------
    nodesRandom : array-like
        Random array for ``nodes``.
    uValues : array-like
        Random array for ``uValues``.
    t_search : float
        Random time to search for.
    t_outside : float
        Random time outside the range.
    index_expected : int
        Index expected for ``t_search`` to be find in ``nodesRandom``.
    """

    import numpy as np

    nodesRandom = {
        0: [
            (0.1, np.array([0.0, 0.0155051, 0.0644949, 0.1])),
            (0.2, np.array([0.1, 0.1155051, 0.1644949, 0.2])),
            (0.3, np.array([0.2, 0.2155051, 0.2644949, 0.3])),
            (0.4, np.array([0.3, 0.3155051, 0.3644949, 0.4])),
        ],
        1: [
            (0.1, np.array([0.0, 0.00571042, 0.0276843, 0.05835904, 0.08602401, 0.1])),
            (0.2, np.array([0.1, 0.10571042, 0.1276843, 0.15835904, 0.18602401, 0.2])),
            (0.3, np.array([0.2, 0.20571042, 0.2276843, 0.25835904, 0.28602401, 0.3])),
            (0.4, np.array([0.3, 0.30571042, 0.3276843, 0.35835904, 0.38602401, 0.4])),
        ],
    }
    uValuesRandom = {
        0: [
            (
                0.1,
                [
                    np.array([[2.0, 0.0], [-0.66666667, 0.0]]),
                    np.array([[1.98961804, 0.0], [-0.67249088, 0.0]]),
                    np.array([[1.95620197, 0.0], [-0.69203802, 0.0]]),
                    np.array([[1.93136107, 0.0], [-0.70741795, 0.0]]),
                ],
            ),
            (
                0.2,
                [
                    np.array([[1.93136107, 0.0], [-0.70741795, 0.0]]),
                    np.array([[1.92033747, 0.0], [-0.71449207, 0.0]]),
                    np.array([[1.88475913, 0.0], [-0.73845028, 0.0]]),
                    np.array([[1.85820566, 0.0], [-0.75754586, 0.0]]),
                ],
            ),
            (
                0.3,
                [
                    np.array([[1.85820566, 0.0], [-0.75754586, 0.0]]),
                    np.array([[1.84639076, 0.0], [-0.76640475, 0.0]]),
                    np.array([[1.80811795, 0.0], [-0.79677676, 0.0]]),
                    np.array([[1.77939743, 0.0], [-0.82141634, 0.0]]),
                ],
            ),
            (
                0.4,
                [
                    np.array([[1.77939743, 0.0], [-0.82141634, 0.0]]),
                    np.array([[1.76657061, 0.0], [-0.83298482, 0.0]]),
                    np.array([[1.72480267, 0.0], [-0.87334246, 0.0]]),
                    np.array([[1.69320979, 0.0], [-0.90693533, 0.0]]),
                ],
            ),
        ],
        1: [
            (
                0.1,
                [
                    np.array([[2.0, 0.0], [-0.66666667, 0.0]]),
                    np.array([[1.99618699, 0.0], [-0.66879257, 0.0]]),
                    np.array([[1.98139946, 0.0], [-0.67718302, 0.0]]),
                    np.array([[1.96044015, 0.0], [-0.68948845, 0.0]]),
                    np.array([[1.94120451, 0.0], [-0.70123256, 0.0]]),
                    np.array([[1.93136107, 0.0], [-0.70741795, 0.0]]),
                ],
            ),
            (
                0.2,
                [
                    np.array([[1.93136107, 0.0], [-0.70741795, 0.0]]),
                    np.array([[1.92731407, 0.0], [-0.70999669, 0.0]]),
                    np.array([[1.91160132, 0.0], [-0.72021219, 0.0]]),
                    np.array([[1.88928009, 0.0], [-0.73530604, 0.0]]),
                    np.array([[1.86873896, 0.0], [-0.74983949, 0.0]]),
                    np.array([[1.85820566, 0.0], [-0.75754586, 0.0]]),
                ],
            ),
            (
                0.3,
                [
                    np.array([[1.85820566, 0.0], [-0.75754586, 0.0]]),
                    np.array([[1.85387058, 0.0], [-0.76076949, 0.0]]),
                    np.array([[1.83701379, 0.0], [-0.77360338, 0.0]]),
                    np.array([[1.81299382, 0.0], [-0.79275739, 0.0]]),
                    np.array([[1.79080725, 0.0], [-0.81142495, 0.0]]),
                    np.array([[1.77939743, 0.0], [-0.82141634, 0.0]]),
                ],
            ),
            (
                0.4,
                [
                    np.array([[1.77939743, 0.0], [-0.82141634, 0.0]]),
                    np.array([[1.77469484, 0.0], [-0.82561542, 0.0]]),
                    np.array([[1.75636993, 0.0], [-0.84245025, 0.0]]),
                    np.array([[1.73014338, 0.0], [-0.86793757, 0.0]]),
                    np.array([[1.70578767, 0.0], [-0.89321743, 0.0]]),
                    np.array([[1.69320901, 0.0], [-0.90693532, 0.0]]),
                ],
            ),
        ],
    }
    t_search = {0: 0.32, 1: 0.19}
    t_outside = {0: 0.43, 1: 0.6}
    index_expected = {0: 3, 1: 1}
    return nodesRandom[case], uValuesRandom[case], t_search[case], t_outside[case], index_expected[case]


def runSimulation(t0, dt, Tend, quad_type, problemType):
    r"""
    Executes a run to solve numerically the Van der Pol with tests to check the ``DenseOutput`` class.

    Parameters
    ----------
    t0 : float
        Initial time.
    dt : float
        Time step size.
    Tend : float
        End time.
    quad_type : str
        Type of quadrature.
    problemType : str
        Type ``'ODE'`` as well as ``'DAE'`` is tested here. Note that only ``newton_tol`` is set, and
        for simulation default parameters of problem classes are used.
    """

    from pySDC.implementations.hooks.log_solution import LogSolution
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    if problemType == 'ODE':
        from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper
        from pySDC.implementations.problem_classes.odeSystem import ProtheroRobinsonAutonomous as problem

    elif problemType == 'DAE':
        from pySDC.projects.DAE.sweepers.semiImplicitDAE import SemiImplicitDAE as sweeper
        from pySDC.projects.DAE.problems.simpleDAE import SimpleDAE as problem

    else:
        raise NotImplementedError(f"For {problemType} no sweeper and problem class is implemented!")

    # initialize level parameters
    level_params = {
        'dt': dt,
        'restol': 1e-12,
    }

    # initialize sweeper parameters
    sweeper_params = {
        'quad_type': quad_type,
        'num_nodes': 3,
        'QI': 'LU',
    }

    problem_params = {
        'newton_tol': 1e-12,
    }

    # initialize step parameters
    step_params = {'maxiter': 5}

    # fill description dictionary for easy step instantiation
    description = {
        'problem_class': problem,
        'problem_params': problem_params,
        'sweeper_class': sweeper,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
    }

    # instantiate controller
    controller_params = {'logger_level': 30, 'hook_class': [LogSolution]}
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    prob = controller.MS[0].levels[0].prob

    uinit = prob.u_exact(t0)

    # call main function to get things done...
    _, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    return prob, stats


@pytest.mark.base
@pytest.mark.parametrize("quad_type", ['RADAU-RIGHT', 'LOBATTO'])
@pytest.mark.parametrize("problemType", ['ODE', 'DAE'])
def test_interpolate(quad_type, problemType):
    r"""
    Interpolation in ``DenseOutput`` is tested by evaluating the polynomials at ``t_eval``.
    The interpolated values are then compared with the reference error.
    """

    import numpy as np
    from pySDC.projects.DAE.misc.denseOutput import DenseOutput
    from pySDC.helpers.stats_helper import get_sorted

    t0 = 0.0
    dt = 1e-2
    Tend = 0.5

    skipTest = True if quad_type == 'LOBATTO' and problemType == 'DAE' else False
    if not skipTest:
        prob, solutionStats = runSimulation(t0=t0, dt=dt, Tend=Tend, quad_type=quad_type, problemType=problemType)
    else:
        prob, solutionStats = None, None

    # get values of u and corresponding times for all nodes
    if solutionStats is not None:
        u_dense = get_sorted(solutionStats, type='u_dense', sortby='time', recomputed=False)
        nodes_dense = get_sorted(solutionStats, type='nodes', sortby='time', recomputed=False)
        sol = DenseOutput(nodes_dense, u_dense)

        t_eval = [t0 + i * 0.05 * dt for i in range(int(Tend / (dt * 0.05)) + 1)]
        u_eval = [sol(t_item) for t_item in t_eval]

        if problemType == 'ODE':
            u_eval = np.array(u_eval)
            uex = np.array([prob.u_exact(t_item) for t_item in t_eval])

        elif problemType == 'DAE':
            x1_eval = np.array([me.diff[0] for me in u_eval])
            x2_eval = np.array([me.diff[1] for me in u_eval])
            z_eval = np.array([me.alg[0] for me in u_eval])

            x1ex = np.array([prob.u_exact(t_item).diff[0] for t_item in t_eval])
            x2ex = np.array([prob.u_exact(t_item).diff[1] for t_item in t_eval])
            zex = np.array([prob.u_exact(t_item).alg[0] for t_item in t_eval])

            u_eval = np.column_stack((np.column_stack((x1_eval, x2_eval)), z_eval))
            uex = np.column_stack((np.column_stack((x1ex, x2ex)), zex))

        for i in range(uex.shape[0]):
            assert np.allclose(u_eval[i, :], uex[i, :], atol=1e-14), f"For index {i} error is too large!"


@pytest.mark.base
@pytest.mark.parametrize("case", [0, 1])
def test_find_time_interval(case):
    """
    Test to check _find_time_interval for some random arrays.
    """

    from pySDC.projects.DAE.misc.denseOutput import DenseOutput

    nodes, uValues, t_search, t_outside, index_expected = getRandomArrays(case)

    sol = DenseOutput(nodes=nodes, uValues=uValues)

    index = sol._find_time_interval(t_search)
    assert index == index_expected, f"Found index wrong! Got {index}, expected {index_expected}!"

    with pytest.raises(ValueError):
        index_outside = sol._find_time_interval(t_outside)


@pytest.mark.base
def test_recover_datatype():
    """Checks if datatype will be recovered. Here: Test for MeshDAE."""

    import numpy as np
    from pySDC.projects.DAE.misc.meshDAE import MeshDAE
    from pySDC.projects.DAE.misc.denseOutput import DenseOutput

    case = 0
    nodes, uValues, _, _, _ = getRandomArrays(case)

    sol = DenseOutput(nodes=nodes, uValues=uValues)
    uValues = sol.uValues

    # convert list to array object
    uValues = np.asarray(uValues)

    # recover datatype - which should be a MeshDAE object
    uRecoverMeshDAE = sol._recover_datatype(uValues, uValues.shape, MeshDAE)
    assert type(uRecoverMeshDAE) == MeshDAE, "Method _recover_datatype does not recover the datatype MeshDAE!"
