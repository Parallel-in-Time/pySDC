import numpy as np
import time

from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.implementations.problem_classes.AllenCahn_1D_FD import allencahn_fullyimplicit
from pySDC.implementations.controller_classes.allinclusive_multigrid_nonMPI import allinclusive_multigrid_nonMPI

from pySDC.playgrounds.Newton_PFASST.allinclusive_jacmatrix_nonMPI import allinclusive_jacmatrix_nonMPI
from pySDC.playgrounds.Newton_PFASST.pfasst_newton_output import output

from pySDC.playgrounds.Newton_PFASST.linear_pfasst.LinearBaseTransfer import linear_base_transfer
from pySDC.playgrounds.Newton_PFASST.linear_pfasst.allinclusive_newton_nonMPI import allinclusive_newton_nonMPI
from pySDC.playgrounds.Newton_PFASST.linear_pfasst.generic_implicit_rhs import generic_implicit_rhs
# from pySDC.playgrounds.Newton_PFASST.linear_pfasst.AllenCahn_1D_FD_jac import allencahn_fullyimplicit, allencahn_fullyimplicit_jac
from pySDC.playgrounds.Newton_PFASST.linear_pfasst.AllenCahn_2D_FD_jac import allencahn_fullyimplicit, allencahn_fullyimplicit_jac

from pySDC.helpers.stats_helper import filter_stats, sort_stats


def setup():
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-08
    level_params['dt'] = 1E-03
    level_params['nsweeps'] = [1]

    # This comes as read-in for the step class (this is optional!)
    step_params = dict()
    step_params['maxiter'] = 50

    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['nu'] = 2
    # problem_params['nvars'] = [128, 64]
    problem_params['nvars'] = [(128, 128), (64, 64)]
    # problem_params['nvars'] = [(512, 512), (256, 256)]
    # problem_params['nvars'] = [(512, 512)]
    problem_params['eps'] = 0.04
    problem_params['newton_maxiter'] = 1
    problem_params['newton_tol'] = 1E-09
    problem_params['lin_tol'] = 1E-08
    problem_params['lin_maxiter'] = 1000
    problem_params['radius'] = 0.25

    # This comes as read-in for the sweeper class
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]
    sweeper_params['QI'] = ['LU']#, 'LU']
    sweeper_params['spread'] = False

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 6
    space_transfer_params['periodic'] = True

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['predict'] = False

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = allencahn_fullyimplicit
    description['problem_params'] = problem_params
    description['dtype_u'] = mesh
    description['dtype_f'] = mesh
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['space_transfer_class'] = mesh_to_mesh
    description['space_transfer_params'] = space_transfer_params

    return description, controller_params


def run_newton_pfasst_matrixfree(Tend=None):

    print('THIS IS MATRIX-FREE NEWTON-PFASST....')

    description, controller_params = setup()

    # setup parameters "in time"
    t0 = 0.0

    num_procs = int((Tend - t0) / description['level_params']['dt'])

    controller_pred = allinclusive_multigrid_nonMPI(num_procs=num_procs, controller_params=controller_params,
                                               description=description)
    # get initial values on finest level
    P = controller_pred.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    controller_pred.params.predict = True
    for S in controller_pred.MS:
        S.params.maxiter = 2

    uend, stats = controller_pred.run(u0=uinit, t0=t0, Tend=Tend)

    uk = [[P.dtype_u(S.levels[0].u[m]) for m in range(1, 4)] for S in controller_pred.MS]
    # uk = np.concatenate([[S.levels[0].u[m] for S in controller_pred.MS for m in
    #                      range(1, description['sweeper_params']['num_nodes'] + 1)])

    description['problem_class'] = allencahn_fullyimplicit_jac
    description['base_transfer_class'] = linear_base_transfer
    description['sweeper_class'] = generic_implicit_rhs
    outer_restol = description['level_params']['restol']
    description['step_params']['maxiter'] = description['problem_params']['newton_maxiter']
    description['level_params']['restol'] = description['problem_params']['newton_tol']

    # instantiate the controller
    controller = allinclusive_newton_nonMPI(num_procs=num_procs, controller_params=controller_params,
                                            description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    uend, stats = controller.run_with_pred(uk=uk, u0=uinit, t0=t0, Tend=Tend)

    # get duration of run
    timing = sort_stats(filter_stats(stats, type='timing_run'), sortby='time')[0][1]

    # compute and print statistics
    nsolves_all = controller.ninnersolve
    nsolves_step = nsolves_all / num_procs
    nsolves_iter = nsolves_all / controller.nouteriter
    print('  --> Number of outer iterations: %i' % controller.nouteriter)
    print('  --> Number of inner solves (total/per iter/per step): %i / %4.2f / %4.2f'
          % (nsolves_all, nsolves_iter, nsolves_step))
    print('  ... took %s sec' % timing)
    print()


def run_pfasst_newton(Tend=None):

    print('THIS IS PFASST-NEWTON....')

    description, controller_params = setup()

    # remove this line to reduce the output of PFASST
    # controller_params['hook_class'] = output

    # setup parameters "in time"
    t0 = 0.0

    num_procs = int((Tend - t0) / description['level_params']['dt'])

    controller = allinclusive_multigrid_nonMPI(num_procs=num_procs, controller_params=controller_params,
                                               description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # get list of iterations count, sorted by process/time
    iter_counts = sort_stats(filter_stats(stats, type='niter'), sortby='time')

    # get maximum number of iterations
    niter = max([item[1] for item in iter_counts])

    # get duration of run
    timing = sort_stats(filter_stats(stats, type='timing_run'), sortby='time')[0][1]

    # compute and print statistics
    nsolves_all = int(np.sum([S.levels[0].prob.newton_itercount for S in controller.MS]))
    nsolves_step = nsolves_all / num_procs
    nsolves_iter = nsolves_all / niter
    print('  --> Number of outer iterations: %i' % niter)
    print('  --> Number of inner solves (total/per iter/per step): %i / %4.2f / %4.2f'
          % (nsolves_all, nsolves_iter, nsolves_step))
    print('  --> took %s sec' % timing)
    print()


def main():

    # Setup can run until 0.032 = 32 * 0.001, so the factor gives the number of time-steps.
    num_procs = 8
    Tend = num_procs * 0.001

    run_newton_pfasst_matrixfree(Tend=Tend)
    run_pfasst_newton(Tend=Tend)


if __name__ == "__main__":

    main()
