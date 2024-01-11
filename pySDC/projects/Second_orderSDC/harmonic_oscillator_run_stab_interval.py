import numpy as np
from pySDC.projects.Second_orderSDC.harmonic_oscillator_params import harmonic_oscillator_params
from pySDC.projects.Second_orderSDC.stability_simulation import compute_and_check_stability

if __name__ == '__main__':
    # Define lists for the number of nodes and maximum iterations
    description = harmonic_oscillator_params()
    helper_params = {
        'quad_type_list': ['GAUSS'],
        'Num_iter': (2000, 1),
        'num_nodes_list': np.arange(2, 7, 1),
        'max_iter_list': np.arange(1, 11, 1),
    }
    description['helper_params'] = helper_params
    points = ((100, 1e-10),)
    # Iterate through points and perform stability check
    for ii in points:
        compute_and_check_stability(description, ii, compute_interval=True)
