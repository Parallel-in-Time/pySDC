
from projects.parallelSDC.nonlinear_playground import main, plot_graphs
from projects.parallelSDC.newton_vs_sdc import main as main_newton_vs_sdc
from projects.parallelSDC.newton_vs_sdc import plot_graphs as plot_graphs_newton_vs_sdc

def test_main():
    main()
    plot_graphs()

def test_newton_vs_sdc():
    main_newton_vs_sdc()
    plot_graphs_newton_vs_sdc()