import os
from projects.FastWaveSlowWave.runconvergence_acoustic import plot_convergence
from projects.FastWaveSlowWave.runitererror_acoustic import compute_and_plot_itererror
from projects.FastWaveSlowWave.runmultiscale_acoustic import compute_and_plot_solutions

def test_plot_convergence():
    assert os.path.isfile('projects/FastWaveSlowWave/conv-data.txt'), 'ERROR: conv-data.txt does not exist'
    plot_convergence(cwd='projects/FastWaveSlowWave/')
    assert os.path.isfile('convergence.png'), 'ERROR: convergence plot has not been created'


def test_compute_and_plot_itererror():
    compute_and_plot_itererror()
    assert os.path.isfile('iteration.png'), 'ERROR: iteration plot has not been created'


def test_compute_and_plot_solutions():
    compute_and_plot_solutions()
    assert os.path.isfile('multiscale-K2-M2.png'), 'ERROR: solution plot has not been created'
