import numpy as np
from pySDC.projects.Second_orderSDC.harmonic_oscillator_params import get_default_harmonic_oscillator_description
from pySDC.projects.Second_orderSDC.stability_simulation import compute_and_generate_table


if __name__ == '__main__':
    '''
    To compute maximum stable values for SDC and Picard iterations for the purely oscillatory case with
    no damping (mu=0)
    Additional parameter:
        To save the table set: save_interval_file=True
        Change filename: interval_filename='FILENAME' by default: './data/stab_interval.txt'
    Output:
        it generates to compare with different values of M (number of nodes) and K (maximal number of iterations)
    '''
    # Ger default parameters
    description = get_default_harmonic_oscillator_description()
    # Additional parameters to compute stability interval on the kappa
    helper_params = {
        'quad_type_list': ('GAUSS',),
        'Num_iter': (2000, 1),
        'num_nodes_list': np.arange(2, 7, 1),
        'max_iter_list': np.arange(1, 11, 1),
    }

    points = ((100, 1e-10),)
    # Iterate through points and perform stability check
    for ii in points:
        compute_and_generate_table(description, helper_params, ii, compute_interval=True)
