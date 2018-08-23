import numpy as np


from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.implementations.problem_classes.AllenCahn_1D_FD import allencahn_fullyimplicit
from pySDC.implementations.controller_classes.allinclusive_multigrid_nonMPI import allinclusive_multigrid_nonMPI

from pySDC.playgrounds.Newton_PFASST.allinclusive_jacmatrix_nonMPI import allinclusive_jacmatrix_nonMPI
from pySDC.playgrounds.Newton_PFASST.pfasst_newton_output import output

from pySDC.helpers.stats_helper import filter_stats, sort_stats


def setup():
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-08
    level_params['dt'] = 1E-03
    level_params['nsweeps'] = 1

    # This comes as read-in for the step class (this is optional!)
    step_params = dict()
    step_params['maxiter'] = 50

    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['nu'] = 2
    problem_params['nvars'] = [128, 64]
    problem_params['eps'] = 0.04
    problem_params['inner_maxiter'] = 1
    problem_params['inner_tol'] = 1E-09
    problem_params['radius'] = 0.25

    # This comes as read-in for the sweeper class
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3, 3]
    sweeper_params['QI'] = ['LU', 'LUinv']

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


def run_newton_pfasst(Tend=None):

    description, controller_params = setup()

    # setup parameters "in time"
    t0 = 0.0

    num_procs = int((Tend - t0) / description['level_params']['dt'])

    # instantiate the controller
    controller = allinclusive_jacmatrix_nonMPI(num_procs=num_procs, controller_params=controller_params,
                                               description=description)

    # print([L.sweep.coll.nodes for L in controller.MS[0].levels])
    # exit()

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    uk = np.kron(np.ones(controller.nsteps * controller.nnodes), uinit.values)

    controller.compute_rhs(uk, t0)
    print(np.linalg.norm(controller.rhs, np.inf))
    k = 0
    while np.linalg.norm(controller.rhs, np.inf) > 1E-08 or k == 0:
        k += 1
        ek, stats = controller.run(uk=uk, t0=t0, Tend=Tend)
        uk -= ek
        controller.compute_rhs(uk, t0)

        print(k, controller.inner_solve_counter, np.linalg.norm(controller.rhs, np.inf), np.linalg.norm(ek, np.inf))

    nsolves_all = controller.inner_solve_counter
    nsolves_step = nsolves_all / num_procs
    print(nsolves_all, nsolves_step)


def run_pfasst_newton(Tend=None):

    description, controller_params = setup()

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

    # # filter statistics by variant (number of iterations)
    # filtered_stats = filter_stats(stats, type='niter')
    #
    # # convert filtered statistics to list of iterations count, sorted by process
    # iter_counts = sort_stats(filtered_stats, sortby='time')

    # compute and print statistics
    nsolves_all = np.sum([S.levels[0].prob.inner_solve_counter for S in controller.MS])
    nsolves_step = nsolves_all / num_procs
    print(nsolves_all, nsolves_step)


if __name__ == "__main__":

    Tend = 4 * 0.001

    run_newton_pfasst(Tend=Tend)
    run_pfasst_newton(Tend=Tend)

