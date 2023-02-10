import pytest


@pytest.mark.base
def test_imex_vs_fully_implicit_leaky_superconductor():
    from pySDC.projects.Resilience.leaky_superconductor import compare_imex_full

    compare_imex_full()
