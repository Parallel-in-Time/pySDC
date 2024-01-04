import numpy as np
from pySDC.projects.Second_orderSDC.penningtrap_Simulation import Stability_implementation
from pySDC.projects.Second_orderSDC.dampedharmonic_oscillator_run_stability import dampedharmonic_oscillator_params
from tabulate import tabulate


def stab_interval(num_nodes_list, max_iter_list, save_file=False, filename='stab_results.txt'):
    # Initialize simulation parameters based on the damped harmonic oscillator
    description = dampedharmonic_oscillator_params()

    # Initialize data as a list of rows
    data = []

    # Loop through different numbers of nodes, maximum iterations, and quadrature types

    for num_nodes in num_nodes_list:
        for max_iter in max_iter_list:
            # Update simulation parameters
            description['sweeper_params']['num_nodes'] = num_nodes
            description['sweeper_params']['quad_type'] = 'GAUSS'
            description['step_params']['maxiter'] = max_iter

            # Create Stability_implementation instance
            stab_model = Stability_implementation(description, kappa_max=100, mu_max=1e-10, Num_iter=(1000, 1))

            # Extract the values where SDC is less than or equal to 1
            mask = stab_model.SDC <= 1 + 1e-14
            for ii in range(len(mask)):
                if mask[ii] == True:
                    kappa_max = stab_model.lambda_kappa[ii]
                else:
                    break

            # Add row to the data
            data.append([num_nodes, max_iter, kappa_max])

    # Define column names
    headers = ["Num Nodes", "Max Iter", 'kappa_max']

    # Print the table using tabulate
    table_str = tabulate(data, headers=headers, tablefmt="grid")

    # Print or save the table to a file
    if save_file:
        with open(filename, 'w') as file:
            file.write(table_str)
        print(f"Table saved to {filename}")
    else:
        print(table_str)


if __name__ == '__main__':
    # Define lists for the number of nodes and maximum iterations
    M_list = np.arange(2, 7, 1)
    K_list = np.arange(1, 11, 1)

    stab_interval(num_nodes_list=M_list, max_iter_list=K_list)
