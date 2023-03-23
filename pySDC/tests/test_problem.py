# Test some functionality of the core problem module
import pytest
import numpy as np


@pytest.mark.base
@pytest.mark.parametrize("init", [[(2, 3, 4)], [(2, 3)], [(1,)]])
def test_scipy_reference(init):
    """
    Test the generation of reference solutions with scipy.
    A Dahlquist problem is solved using scipy and exactly. Depending on the shape that is passed in `init`, this can
    emulate a PDE. What is really tested in terms of PDEs is that the changes in shape of the solution object is handled
    correctly.

    Args:
        init (list): Object similar to the `init` that you use for the problem class

    Returns:
        None
    """
    from pySDC.core.Problem import ptype

    # instantiate a dummy problem
    problem = ptype(init)

    # setup random initial conditions
    u0 = np.random.rand(*init[0])
    lamdt = np.random.rand(*u0.shape)

    # define function to evaluate the right hand side
    def eval_rhs(t, u):
        return (u.reshape(init[0]) * -lamdt).flatten()

    # compute two solutions: One with scipy and one analytic exact solution
    u_ref = problem.generate_scipy_reference_solution(eval_rhs, 1.0, u_init=u0.copy(), t_init=0)
    u_exact = u0 * np.exp(-lamdt)

    # check that the two solutions are the same to high degree
    assert (
        u_ref.shape == u_exact.shape
    ), "The shape of the scipy reference solution does not match the shape of the actual solution"
    assert np.allclose(u_ref, u_exact, atol=1e-12), "The scipy solution deviates significantly from the exact solution"


class TestBasics:
    # To avoid clash between dolfin and mpi4py, need to import dolfin first
    try:
        import dolfin as df
    except ImportError:
        pass
    finally:
        del df

    from pySDC.implementations.problem_classes.LogisticEquation import logistics_equation

    PROBLEMS = {
        logistics_equation: {
            'probParams': dict(u0=2.0, newton_maxiter=100, newton_tol=1e-6, direct=True, lam=0.5, stop_at_nan=True),
            'testParams': {'tBeg': 0, 'tEnd': 1.0, 'nSteps': 1000, 'tol': 1e-3},
        }
    }

    @pytest.mark.base
    @pytest.mark.parametrize('probType', PROBLEMS.keys())
    def test_uExact_accuracy(self, probType):
        params = self.PROBLEMS[probType]['probParams']
        prob = probType(**params)

        testParams = self.PROBLEMS[probType]['testParams']
        tBeg = testParams['tBeg']
        tEnd = testParams['tEnd']
        nSteps = testParams['nSteps']
        dt = (tEnd - tBeg) / nSteps
        uNum = prob.u_exact(tBeg)
        for n in range(nSteps):
            uNum = uNum + dt * prob.eval_f(uNum, tBeg + n * dt)

        assert np.linalg.norm(prob.u_exact(tEnd) - uNum, ord=np.inf) < testParams['tol']


if __name__ == '__main__':
    test_scipy_reference([(2, 3)])

    from pySDC.implementations.problem_classes.LogisticEquation import logistics_equation

    prob = TestBasics()
    prob.test_uExact_accuracy(logistics_equation)
