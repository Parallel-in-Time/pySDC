from pySDC import Log
from pySDC.Level import level
from pySDC.Hooks import hooks

from implementations.problem_classes.HeatEquation_1D_FD_forced import heat1d_forced
from implementations.datatype_classes.mesh import mesh, rhs_imex_mesh
from implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from implementations.sweeper_classes.imex_1st_order import imex_1st_order

if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    # initialize level parameters
    level_params = {}
    level_params['restol'] = 1E-10
    level_params['dt'] = 0.1

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3

    # initialize problem parameters
    problem_params = {}
    problem_params['nu'] = 0.1  # diffusion coefficient
    problem_params['freq'] = 4  # frequency for the test value
    problem_params['nvars'] = 1023  # number of degrees of freedom

    L = level(problem_class  = heat1d_forced,
              problem_params = problem_params,
              dtype_u        = mesh,
              dtype_f        = rhs_imex_mesh,
              sweeper_class  = imex_1st_order,
              sweeper_params = sweeper_params,
              level_params   = level_params,
              hook_class     = hooks,
              id             = "imextest")

    P = L.prob
    L.status.time = 0.0
    L.u[0] = P.u_exact(L.time)

    L.sweep.predict()
    L.sweep.compute_residual()
    k = 0
    max_iter = 20
    while k < max_iter and L.status.residual > L.params.restol:
        k += 1
        L.sweep.update_nodes()
        L.sweep.compute_residual()
        logger.info('time %8.6f of %s -- Iteration: %2i -- Residual: %12.8e',
                    L.time, L.id, k, L.status.residual)

    print(L.status.residual,k)
    L.sweep.compute_end_point()

    # compute exact solution and compare
    uex = P.u_exact(L.dt)

    print('error at time %s: %s' %(L.dt,abs(uex-L.uend)))