import os
from projects.FastWaveSlowWave.runconvergence_acoustic import plot_convergence

def test_plot_convergence():
    assert os.path.isfile('projects/FastWaveSlowWave/conv-data.txt'), 'ERROR: conv-data.txt does not exist'
    plot_convergence(cwd='projects/FastWaveSlowWave/')
    assert os.path.isfile('convergence.png'), 'ERROR: convergence plot has not been created'


