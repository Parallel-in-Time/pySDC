from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced import fenics_heat_mass
from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
from pySDC.implementations.transfer_classes.BaseTransfer_mass import base_transfer_mass
from pySDC.implementations.transfer_classes.TransferFenicsMesh import mesh_to_mesh_fenics

from pySDC.playgrounds.FEniCS.HookClass_FEniCS_output import fenics_output

from pySDC.helpers.stats_helper import filter_stats, sort_stats

if __name__ == "__main__":
    num_procs = 1

    t0 = 0
    dt = 0.2
    Tend = 0.2

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-10
    level_params['dt'] = dt

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]

    problem_params = dict()
    problem_params['nu'] = 0.1
    problem_params['t0'] = t0  # ugly, but necessary to set up ProblemClass
    problem_params['c_nvars'] = [128]
    problem_params['family'] = 'CG'
    problem_params['order'] = [1]
    problem_params['refinements'] = [1, 0]

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = fenics_output

    base_transfer_params = dict()
    # base_transfer_params['finter'] = True

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = fenics_heat_mass
    description['problem_params'] = problem_params
    description['sweeper_class'] = imex_1st_order_mass
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['space_transfer_class'] = mesh_to_mesh_fenics
    description['base_transfer_class'] = base_transfer_mass
    description['base_transfer_params'] = base_transfer_params

    # quickly generate block of steps
    controller = controller_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # compute exact solution and compare
    uex = P.u_exact(Tend)

    print('(classical) error at time %s: %s' % (Tend, abs(uex - uend) / abs(uex)))

    errors = sort_stats(filter_stats(stats, type='error'), sortby='iter')
    residuals = sort_stats(filter_stats(stats, type='residual'), sortby='iter')

    for err, res in zip(errors, residuals):
        print(err[0], err[1], res[1])
