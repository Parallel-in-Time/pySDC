import numpy as np
from pySDC.projects.Second_orderSDC.penningtrap_Simulation import Stability_implementation
from pySDC.projects.Second_orderSDC.dampedharmonic_oscillator_run_stability import dampedharmonic_oscillator_params
from tabulate import tabulate


def check_stab_points(points, num_nodes_list, max_iter_list, store=False, filename='stab_results.txt'):
    # Initialize simulation parameters based on the damped harmonic oscillator
    description = dampedharmonic_oscillator_params()
    quad_type_list = ['GAUSS', 'LOBATTO', 'RADAU-LEFT']

    # Initialize data as a list of rows
    data = []

    # Loop through different numbers of nodes and maximum iterations
    for quad_type in quad_type_list:
        for num_nodes in num_nodes_list:
            for max_iter in max_iter_list:
                # Update simulation parameters
                description['sweeper_params']['num_nodes'] = num_nodes
                description['sweeper_params']['quad_type'] = quad_type
                description['step_params']['max_iter'] = max_iter

                # Create Stability_implementation instance
                stab_model = Stability_implementation(
                    description, kappa_max=points[0], mu_max=points[1], Num_iter=(2, 2)
                )

                # Check stability and print results
                if stab_model.SDC[-1, -1] <= 1:
                    stability_result = "Stable"
                else:
                    stability_result = "Unstable. Increase M or K"

                # Add row to the data
                data.append([quad_type, num_nodes, max_iter, points, stability_result, stab_model.SDC[-1, -1]])

    # Define column names
    headers = ["Quad Type", "Num Nodes", "Max Iter", "(kappa, mu)", "Stability", "Stability Radius"]

    # Print the table using tabulate
    table_str = tabulate(data, headers=headers, tablefmt="grid")

    # Print or save the table to a file
    if store:
        with open(filename, 'w') as file:
            file.write(table_str)
        print(f"Table saved to {filename}")
    else:
        print(table_str)


if __name__ == '__main__':
    # Define lists for the number of nodes and maximum iterations
    M_list = np.arange(3, 6, 1)
    K_list = np.arange(2, 10, 1)

    # Define points for stability check
    points = ((3, 4), (3, 6), (10, 60))

    # Iterate through points and perform stability check
    for ii in points:
        check_stab_points(ii, num_nodes_list=M_list, max_iter_list=K_list)
