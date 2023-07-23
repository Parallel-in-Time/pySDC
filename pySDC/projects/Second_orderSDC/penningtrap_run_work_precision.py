from pySDC.projects.Second_orderSDC.penningtrap_Simulation import compute_error
from pySDC.projects.Second_orderSDC.penningtrap_run_error import penningtrap_param

if __name__ == '__main__':
    """
    Convergence plot for the second order SDC:
        K_iter: The number of iterations
        time_iter: the number of time slices in the time/2**time_iter
        axes: Axis to show the plot
        RKN True or False: To implement together RKN method
        VV True or False: To implement Velocity-Verlet scheme
    """
    exec(open("check_data_folder.py").read())
    Tend = 128 * 0.015625
    controller_params, description = penningtrap_param()
    # description['level_params']['dt'] = 0.015625 * 2
    work_pre = compute_error(controller_params, description, time_iter=3, Tend=Tend, K_iter=(2, 4, 6), axes=(0,))
    work_pre.run_work_precision()
