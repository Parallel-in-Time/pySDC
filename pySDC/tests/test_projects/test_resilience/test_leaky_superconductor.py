import pytest


@pytest.mark.base
def test_imex_vs_fully_implicit_leaky_superconductor():
    """
    Test if the IMEX and fully implicit schemes get the same solution and that the runaway process has started.
    """
    from pySDC.projects.Resilience.leaky_superconductor import compare_imex_full

    compare_imex_full()
