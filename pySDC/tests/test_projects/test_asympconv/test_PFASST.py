import pytest


@pytest.mark.base
def test_main():
    from pySDC.projects.AsympConv.PFASST_conv_tests import main

    main()


@pytest.mark.base
def test_plot_results():
    from pySDC.projects.AsympConv.PFASST_conv_Linf import plot_results

    plot_results(cwd='pySDC/projects/AsympConv/')
