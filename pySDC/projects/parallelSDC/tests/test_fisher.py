import pytest


@pytest.mark.base
def test_main():
    from pySDC.projects.parallelSDC.nonlinear_playground import main, plot_graphs

    main()
    plot_graphs()


@pytest.mark.base
def test_newton_vs_sdc():
    from pySDC.projects.parallelSDC.newton_vs_sdc import main as main_newton_vs_sdc
    from pySDC.projects.parallelSDC.newton_vs_sdc import plot_graphs as plot_graphs_newton_vs_sdc

    main_newton_vs_sdc()
    plot_graphs_newton_vs_sdc()
