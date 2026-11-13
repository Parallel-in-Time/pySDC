import numpy as np
import pytest


@pytest.mark.base
def test_bridge_converges_to_the_piecewise_linear_path():
    """The smooth bridge matches the path at the ends of the step, and converges to it between them as J grows."""
    from pySDC.projects.SDC4SDEs.brownian import bridge, bridge_coefficients, brownian_paths

    W, dt = brownian_paths(5, 64, T=0.5, seed=1)[0], 0.5
    s = np.linspace(0, dt, 65)
    previous = np.inf
    for J in [0, 16, 256, 4096]:
        eta0, xi = bridge_coefficients(W, dt, J)
        B = bridge(eta0, xi, dt, s)
        assert np.allclose(B[:, [0, -1]], W[:, [0, -1]] - W[:, :1], atol=1e-12)
        error = np.abs(B - (W - W[:, :1])).max()
        assert error < previous
        previous = error
    assert previous < 1e-3


@pytest.mark.base
def test_lagrange_cosine_matrices():
    """The integrals of the Lagrange polynomials times cosines are exact for polynomials up to degree M - 1."""
    from pySDC.core.collocation import CollBase
    from pySDC.projects.SDC4SDEs.sweeper import lagrange_cosine_matrices

    tau = CollBase(num_nodes=4, quad_type='LOBATTO').nodes
    Q = lagrange_cosine_matrices(tau, 40)
    for k in [1, 7, 40]:
        w = k * np.pi
        # integral of s^2 cos(w s) from 0 to tau
        exact = ((w**2 * tau**2 - 2) * np.sin(w * tau) + 2 * w * tau * np.cos(w * tau)) / w**3
        assert np.allclose(Q[k - 1, 1:, 1:] @ tau**2, exact, atol=1e-14)
        assert np.allclose(Q[k - 1, 1:, 1:].sum(axis=1), np.sin(w * tau) / w, atol=1e-14)


@pytest.mark.base
def test_sdc_converges_to_the_random_ode():
    """
    With enough nodes and sweeps, SDC-SBB gives the exact solution of the random ODE, cosine terms included. The
    collocation error converges in the number of nodes, not in the step size, whose bridge frequencies scale with it.
    """
    from pySDC.projects.SDC4SDEs.brownian import brownian_paths
    from pySDC.projects.SDC4SDEs.problem_classes import OrnsteinUhlenbeck
    from pySDC.projects.SDC4SDEs.run import solve_sdc

    params = {'brownian_path': brownian_paths(20, 256, seed=2), 'n_bridge_terms': 2}
    u, P = solve_sdc(OrnsteinUhlenbeck, params, 0.25, num_nodes=10, num_sweeps=40, initial_guess='spread')
    assert np.abs(u - P.solve_sbb_exactly(0.25, 1.0)).max() < 1e-12


@pytest.mark.base
@pytest.mark.parametrize('problem', ['linear', 'nonlinear', 'system'])
def test_orders(problem):
    """
    SDC-M on M Lobatto nodes with 2(M - 1) sweeps converges with order M - 1, Euler-Maruyama 1/2, Milstein 1. The
    bounds allow for the fits on few, large steps. Explicit Euler-Maruyama and Milstein are left out for the nonlinear
    SDE, where they blow up on some paths for these steps (the paper uses implicit versions there).
    """
    from pySDC.projects.SDC4SDEs.studies import convergence_and_work
    from pySDC.projects.SDC4SDEs.run import order

    dts = [2.0**-k for k in range(3, 7)]
    result = convergence_and_work(problem, n_samples=400, dts=dts, nodes=(3, 4), seed=0, plot=False)
    expected = (
        {'SDC-3': 2, 'SDC-4': 3}
        if problem == 'nonlinear'
        else {'Euler-Maruyama': 0.5, 'Milstein': 1, 'SDC-3': 2, 'SDC-4': 3}
    )
    for method, p in expected.items():
        for i in range(result[method]['strong'].shape[1]):
            assert p - 0.5 < order(dts, result[method]['strong'][:, i]) < p + 0.6, (method, i)


@pytest.mark.base
def test_two_sweeps_per_order():
    """On the linear SDE, every second sweep gains one order (Figure 1 of the paper)."""
    from pySDC.projects.SDC4SDEs.studies import sweeps_and_order
    from pySDC.projects.SDC4SDEs.run import order

    result = sweeps_and_order(n_samples=400, max_sweeps=6, seed=0, plot=False)
    for initial_guess, first in [('euler', 0.0), ('euler_maruyama', 0.5)]:
        orders = [order(result['dt'], e) for e in result[initial_guess]]
        assert abs(orders[0] - first) < 0.15
        for k in [2, 4, 6]:
            assert k / 2 - 0.2 < orders[k] < k / 2 + 0.3, (initial_guess, k, orders)


@pytest.mark.base
@pytest.mark.parametrize(
    'initial_guess, predictor_work', [('spread', 2 * 4), ('euler', 2 * 4), ('euler_maruyama', 2 * 4 + 2)]
)
def test_work(initial_guess, predictor_work):
    """Work per sample and step: drift and diffusion at the left end and at each further node, per sweep."""
    from pySDC.projects.SDC4SDEs.brownian import brownian_paths
    from pySDC.projects.SDC4SDEs.problem_classes import LinearScalarSDE
    from pySDC.projects.SDC4SDEs.run import solve

    params = {'brownian_path': brownian_paths(3, 8, seed=5)}
    _, P = solve({'num_nodes': 4, 'num_sweeps': 2, 'initial_guess': initial_guess}, LinearScalarSDE, params, 0.25)
    assert sum(c.niter for c in P.work_counters.values()) == 4 * (predictor_work + 2 * 2 * 3)
    for method, per_step in [('EM', 2), ('Milstein', 3)]:
        _, P = solve(method, LinearScalarSDE, params, 0.25)
        assert sum(c.niter for c in P.work_counters.values()) == 4 * per_step


@pytest.mark.base
def test_invalid_setups():
    from pySDC.core.errors import ParameterError
    from pySDC.projects.SDC4SDEs.brownian import brownian_paths
    from pySDC.projects.SDC4SDEs.problem_classes import LinearScalarSDE, LinearSystemSDE
    from pySDC.projects.SDC4SDEs.run import solve_sdc

    W = brownian_paths(2, 8, seed=6)
    with pytest.raises(ParameterError):
        solve_sdc(LinearScalarSDE, {'brownian_path': W}, 0.25, 3, 1, initial_guess='zero')
    with pytest.raises(ValueError, match='not on the grid'):
        solve_sdc(LinearScalarSDE, {'brownian_path': W}, 0.1, 3, 1)
    with pytest.raises(ValueError, match='expected paths of shape'):
        LinearSystemSDE(W)
    with pytest.raises(ValueError, match='commute'):
        LinearSystemSDE(np.concatenate([W, W]), B=[np.eye(2), np.array([[0, 1], [0, 0]])])


@pytest.mark.base
def test_ornstein_uhlenbeck():
    """
    The paper's Figure 7: on the smooth path, SDC-SBB converges to the smooth reference far below the distance between
    the smooth and the rough reference; on the rough path itself, it converges with order 1 only.
    """
    import os

    from pySDC.projects.SDC4SDEs.studies import ornstein_uhlenbeck
    from pySDC.projects.SDC4SDEs.run import order

    result = ornstein_uhlenbeck(n_samples=100, n_fine=2**12, n_bridge_terms=64, dts=[2.0**-k for k in range(2, 8)])
    floor = result['random ODE, J = 64, vs. rough path']
    assert np.ptp(floor) < 0.1 * floor.min()
    assert result['SDC-4, J = 64, vs. smooth path'][-1] < 1e-3 * floor.min()
    assert result['random ODE, J = 64, vs. smooth path'].max() < 1e-2 * floor.min()
    for J in [0, 64]:
        assert abs(order(result['dt'], result[f'SDC-4 on rough path, J = {J}']) - 1) < 0.2
    assert os.path.isfile(result['plot'])


@pytest.mark.base
def test_plots():
    """The plots of the studies, for the website."""
    import os

    from pySDC.projects.SDC4SDEs.studies import PROBLEMS, convergence_and_work, sweeps_and_order

    assert os.path.isfile(sweeps_and_order(n_samples=100, max_sweeps=4)['plot'])
    for problem in PROBLEMS:
        dts = [2.0**-k for k in range(2, 7)]
        assert os.path.isfile(convergence_and_work(problem, n_samples=100, dts=dts)['plot'])
