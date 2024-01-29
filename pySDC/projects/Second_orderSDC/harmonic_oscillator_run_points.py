import numpy as np
from pySDC.projects.Second_orderSDC.harmonic_oscillator_params import get_default_harmonic_oscillator_description
from pySDC.projects.Second_orderSDC.stability_simulation import compute_and_generate_table


if __name__ == '__main__':
    '''
    This script generates table for the given points to compare in what quadrature type,
    number of nodes and number of iterations the SDC iteration is stable or not.
    Additional parameter:
        To save the table set: save_points_table=True
        Change filename: points_table_filename='FILENAME' by default: './data/point_table.txt'
    '''
    # Get default parameters for the harmonic osicillator problem
    description = get_default_harmonic_oscillator_description()
    # Additonal params to compute stability points
    helper_params = {
        'quad_type_list': ('GAUSS', 'LOBATTO'),
        'Num_iter': (2, 2),
        'num_nodes_list': np.arange(3, 6, 1),
        'max_iter_list': np.arange(2, 10, 1),
    }

    points = ((1, 100), (3, 100), (10, 100))
    # Iterate through points and perform stability check
    for ii in points:
        compute_and_generate_table(description, helper_params, ii, check_stability_point=True)
