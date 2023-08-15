# It checks whether data folder exicits or not
exec(open("check_data_folder.py").read())

from pySDC.projects.Second_orderSDC.penningtrap_Simulation import compute_error
from pySDC.projects.Second_orderSDC.penningtrap_params import penningtrap_params

if __name__ == '__main__':
    """
    Work-precision implementation for the Second-order SDC
    All parameters are given in penningtrap_params
    Note:
        * time needs to be changed according to choosen axis
        * Tend fixed but it can be changed by defining Tend and include in it in compute_error
        * To implement Velocity-Verlet scheme set VV=True like run_work_precision(VV=True)
        * RKN method can be removed by setting RKN=False like run_work_precision(RKN=False)
        * dt timestep can be changed here as well
        * Make sure dt_cont is setted manually to get suitable picture in work precision
        * it moves RKN and VV line left and right to get right position with SDC and Picard iterations
    """
    controller_params, description = penningtrap_params()
    ## =============================================================================
    ##     dt-timestep and Tend can be changed here manually
    Tend = 128 * 0.015625
    description['level_params']['dt'] = 0.015625 * 4
    description['sweeper_params']['initial_guess'] = 'spread'  # 'zero', 'spread'
    ## =============================================================================
    work_pre = compute_error(controller_params, description, time_iter=3, Tend=Tend, K_iter=(1, 2, 3), axes=(2,))
    work_pre.run_work_precision(RK=True)
