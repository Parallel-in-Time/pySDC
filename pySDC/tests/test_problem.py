# Test some functionality of the core problem module
import pytest


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
    import numpy as np
    from pySDC.core.Problem import ptype

    # instantiate a dummy problem
    problem = ptype(init, None, None)

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


if __name__ == '__main__':
    test_scipy_reference([(2, 3)])
