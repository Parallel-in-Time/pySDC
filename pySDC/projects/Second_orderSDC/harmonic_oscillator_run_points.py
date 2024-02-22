import numpy as np
from pySDC.projects.Second_orderSDC.harmonic_oscillator_params import get_default_harmonic_oscillator_description
from pySDC.projects.Second_orderSDC.stability_simulation import compute_and_generate_table

if __name__ == '__main__':
    '''
    This script generates a table to compare stability of the SDC iteration for given points,
    exploring different quadrature types, number of nodes, and number of iterations.

    Additional parameters in the function `compute_and_generate_table`:
        - To save the table, set: `save_points_table=True`.
        - To change the filename, set `points_table_filename='FILENAME'`. Default is './data/point_table.txt'.
    '''
    # This code checks if the "data" folder exists or not.
    exec(open("check_data_folder.py").read())
    # Get default parameters for the harmonic oscillator problem
    description = get_default_harmonic_oscillator_description()

    # Additional parameters to compute stability points
    helper_params = {
        'quad_type_list': ('GAUSS', 'LOBATTO'),  # List of quadrature types
        'Num_iter': (2, 2),  # Number of iterations
        'num_nodes_list': np.arange(3, 6, 1),  # List of number of nodes
        'max_iter_list': np.arange(2, 10, 1),  # List of maximum iterations
    }

    points = ((1, 100), (3, 100), (10, 100))  # Stability parameters: (kappa, mu)

    # Iterate through points and perform stability check
    for ii in points:
        compute_and_generate_table(description, helper_params, ii, check_stability_point=True, save_points_table=False)
