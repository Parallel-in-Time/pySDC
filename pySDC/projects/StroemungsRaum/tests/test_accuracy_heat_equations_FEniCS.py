import pytest


@pytest.mark.fenics
@pytest.mark.mpi4py
@pytest.mark.slow
def test_3rd_order():

    from pySDC.projects.StroemungsRaum.run_accuracy.run_accuracy_heat_equation_FEniCS import run_accuracy

    # parameters for 3rd order accuracy test
    c_nvars = 64  # spatial resolution
    num_nodes = 2  # number of collocation nodes in time
    p = 2 * num_nodes - 1  # expected order of accuracy

    # Run the simulation
    _, order = run_accuracy(c_nvars, num_nodes)

    # Check if the observed order is close to the expected order
    assert abs(order - p) < 0.1, f"Expected order of accuracy around {p}, but observed {order:.2f}"
