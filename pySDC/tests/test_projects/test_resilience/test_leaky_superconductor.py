import pytest


@pytest.mark.base
@pytest.mark.parametrize('leak_type', ['linear', 'exponential'])
def test_imex_vs_fully_implicit_leaky_superconductor(leak_type):
    """
    Test if the IMEX and fully implicit schemes get the same solution and that the runaway process has started.
    """
    from pySDC.projects.Resilience.leaky_superconductor import compare_imex_full

    compare_imex_full(plotting=False, leak_type=leak_type)
