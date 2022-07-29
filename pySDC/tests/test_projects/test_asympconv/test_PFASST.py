from pySDC.projects.AsympConv.PFASST_conv_Linf import plot_results
from pySDC.projects.AsympConv.PFASST_conv_tests import main


def test_main():
    main()


def test_plot_results():
    plot_results(cwd='pySDC/projects/AsympConv/')
