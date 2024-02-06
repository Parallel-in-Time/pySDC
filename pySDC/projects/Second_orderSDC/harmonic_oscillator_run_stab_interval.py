import numpy as np
from pySDC.projects.Second_orderSDC.harmonic_oscillator_params import get_default_harmonic_oscillator_description
from pySDC.projects.Second_orderSDC.stability_simulation import compute_and_generate_table

if __name__ == '__main__':
    '''
    Main script to compute maximum stable values for SDC and Picard iterations for the purely oscillatory case with no damping (mu=0).

    Additional parameters in the `compute_and_generate_table` function:
        - To save the stability table, set `save_interval_file=True`.
        - To change the filename, set `interval_filename='FILENAME'`. Default is './data/stab_interval.txt'.

    Output:
        The script generates data to compare different values of M (number of nodes) and K (maximal number of iterations).
    '''
    # This code checks if the "data" folder exists or not.
    exec(open("check_data_folder.py").read())
    # Get default parameters for the harmonic oscillator
    description = get_default_harmonic_oscillator_description()

    # Additional parameters to compute stability interval on the kappa
    # =============================================================================
    #     To get exactly the same as table in the paper set:
    #       'Num_iter': (500, 1) for the SDC iteration
    #       'Num_iter': (2000, 1) for the Picard iteration
    # =============================================================================
    helper_params = {
        'quad_type_list': ('GAUSS',),  # Type of quadrature
        'Num_iter': (500, 1),  # Number of iterations
        'num_nodes_list': np.arange(2, 7, 1),  # List of number of nodes
        'max_iter_list': np.arange(1, 11, 1),  # List of maximum iterations
    }

    points = ((100, 1e-10),)  # Stability parameters: (Num_iter, mu)

    # Iterate through points and perform stability check
    for ii in points:
        # If you want to get the table for the Picard iteration set Picard=True
        compute_and_generate_table(
            description, helper_params, ii, compute_interval=True, Picard=False, save_interval_file=True
        )
