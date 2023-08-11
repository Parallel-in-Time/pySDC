from pySDC.projects.Second_orderSDC.penningtrap_params import penningtrap_params
from pySDC.projects.Second_orderSDC.penningtrap_Simulation import compute_error

if __name__ == '__main__':
    """
    Implementation of the order plots of the Spectral deferred correction method for second-order problems paper.
    Both local and global convergence-order plots.
    To implement local convergence plot:
        Run: conv.run_local_error()
    To implement global convergence plot:
        Run: conv.run_global_error()
    Data:
        All of the data and plots saved to the data folder

    Note:
        Tend: final time value can be given manually by default Tend=2
    """
    # Get params for the penning trap problem from the function
    controller_params, description = penningtrap_params()
## =============================================================================
##     dt-timestep can be changed here manually
    description['level_params']['dt']= 0.015625
## =============================================================================
    # Give the parameters to the class
    conv = compute_error(controller_params, description, time_iter=3, K_iter=(1, 2, 3), axes=(0,))
    # Run local convergence order
    # conv.run_local_error()
    # Run global convergence order
# =============================================================================
#     To find apporximate order and expected order you can use this function
#     it is going to save the values in data/global_order_vs_approxorder.csv file
# =============================================================================
    conv.run_global_error()
