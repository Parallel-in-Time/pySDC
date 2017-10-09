from projects.AsympConv.PFASST_conv_tests import main
from projects.AsympConv.PFASST_conv_Linf import plot_results


def test_main():
    main()

def test_plot_results():
    plot_results(cwd='projects/AsympConv/')

