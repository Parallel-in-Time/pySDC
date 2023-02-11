import numpy as np
from pathlib import Path

from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_forced
from pySDC.implementations.problem_classes.AdvectionEquation_ND_FD import advectionNd
from pySDC.implementations.problem_classes.Auzinger_implicit import auzinger
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.implementations.transfer_classes.TransferMesh_NoCoarse import mesh_to_mesh as mesh_to_mesh_nc
from pySDC.implementations.convergence_controller_classes.check_iteration_estimator import CheckIterationEstimatorNonMPI
from pySDC.tutorial.step_8.HookClass_error_output import error_output


def setup_diffusion(dt=None, ndim=None, ml=False):
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-10
    level_params['dt'] = dt  # time-step size
    level_params['nsweeps'] = 1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = ['LU']  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    # sweeper_params['initial_guess'] = 'zero'

    # initialize problem parameters
    problem_params = dict()
    problem_params['ndim'] = ndim  # will be iterated over
    problem_params['order'] = 8  # order of accuracy for FD discretization in space
    problem_params['nu'] = 0.1  # diffusion coefficient
    problem_params['bc'] = 'periodic'  # boundary conditions
    problem_params['freq'] = tuple(2 for _ in range(ndim))  # frequencies
    if ml:
        problem_params['nvars'] = [tuple(64 for _ in range(ndim)), tuple(32 for _ in range(ndim))]  # number of dofs
    else:
        problem_params['nvars'] = tuple(64 for _ in range(ndim))  # number of dofs
    problem_params['direct_solver'] = False  # do GMRES instead of LU
    problem_params['liniter'] = 10  # number of GMRES iterations

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50
    step_params['errtol'] = 1e-07

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 6
    space_transfer_params['periodic'] = True

    # setup the iteration estimator
    convergence_controllers = dict()
    convergence_controllers[CheckIterationEstimatorNonMPI] = {'errtol': 1e-7}

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = error_output

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = heatNd_forced  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['convergence_controllers'] = convergence_controllers
    if ml:
        description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class
        description['space_transfer_params'] = space_transfer_params  # pass parameters for spatial transfer

    return description, controller_params


def setup_advection(dt=None, ndim=None, ml=False):
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-10
    level_params['dt'] = dt  # time-step size
    level_params['nsweeps'] = 1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = ['LU']  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    # sweeper_params['initial_guess'] = 'zero'

    # initialize problem parameters
    problem_params = dict()
    problem_params['ndim'] = ndim  # will be iterated over
    problem_params['order'] = 6  # order of accuracy for FD discretization in space
    problem_params['type'] = 'center'  # order of accuracy for FD discretization in space
    problem_params['bc'] = 'periodic'  # boundary conditions
    problem_params['c'] = 0.1  # diffusion coefficient
    problem_params['freq'] = tuple(2 for _ in range(ndim))  # frequencies
    if ml:
        problem_params['nvars'] = [tuple(64 for _ in range(ndim)), tuple(32 for _ in range(ndim))]  # number of dofs
    else:
        problem_params['nvars'] = tuple(64 for _ in range(ndim))  # number of dofs
    problem_params['direct_solver'] = False  # do GMRES instead of LU
    problem_params['liniter'] = 10  # number of GMRES iterations

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50
    step_params['errtol'] = 1e-07

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 6
    space_transfer_params['periodic'] = True

    # setup the iteration estimator
    convergence_controllers = dict()
    convergence_controllers[CheckIterationEstimatorNonMPI] = {'errtol': 1e-7}

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = error_output

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = advectionNd
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['convergence_controllers'] = convergence_controllers
    if ml:
        description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class
        description['space_transfer_params'] = space_transfer_params  # pass parameters for spatial transfer

    return description, controller_params


def setup_auzinger(dt=None, ml=False):
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-10
    level_params['dt'] = dt  # time-step size
    level_params['nsweeps'] = 1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    if ml:
        sweeper_params['num_nodes'] = [3, 2]
    else:
        sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = ['LU']  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    # sweeper_params['initial_guess'] = 'zero'

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-12
    problem_params['newton_maxiter'] = 10

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50
    step_params['errtol'] = 1e-07

    # setup the iteration estimator
    convergence_controllers = dict()
    convergence_controllers[CheckIterationEstimatorNonMPI] = {'errtol': 1e-7}

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = error_output

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = auzinger
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['convergence_controllers'] = convergence_controllers
    if ml:
        description['space_transfer_class'] = mesh_to_mesh_nc  # pass spatial transfer class

    return description, controller_params


def run_simulations(type=None, ndim_list=None, Tend=None, nsteps_list=None, ml=False, nprocs=None):
    """
    A simple test program to do SDC runs for the heat equation in various dimensions
    """

    t0 = None
    dt = None
    description = None
    controller_params = None

    Path("data").mkdir(parents=True, exist_ok=True)
    f = open('data/step_8_C_out.txt', 'a')

    for ndim in ndim_list:
        for nsteps in nsteps_list:
            if type == 'diffusion':
                # set time parameters
                t0 = 0.0
                dt = (Tend - t0) / nsteps
                description, controller_params = setup_diffusion(dt, ndim, ml)
                mean_number_of_iterations = 3.00 if ml else 5.75
            elif type == 'advection':
                # set time parameters
                t0 = 0.0
                dt = (Tend - t0) / nsteps
                description, controller_params = setup_advection(dt, ndim, ml)
                mean_number_of_iterations = 2.00 if ml else 4.00
            elif type == 'auzinger':
                assert ndim == 1
                # set time parameters
                t0 = 0.0
                dt = (Tend - t0) / nsteps
                description, controller_params = setup_auzinger(dt, ml)
                mean_number_of_iterations = 3.62 if ml else 5.62

            out = f'Running {type} in {ndim} dimensions with time-step size {dt}...\n'
            f.write(out + '\n')
            print(out)

            # Warning: this is black magic used to run an 'exact' collocation solver for each step within the hooks
            description['step_params']['description'] = description
            description['step_params']['controller_params'] = controller_params

            # instantiate controller
            controller = controller_nonMPI(
                num_procs=nprocs, controller_params=controller_params, description=description
            )

            # get initial values on finest level
            P = controller.MS[0].levels[0].prob
            uinit = P.u_exact(t0)

            # call main function to get things done...
            uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

            # filter statistics by type (number of iterations)
            iter_counts = get_sorted(stats, type='niter', sortby='time')

            niters = np.array([item[1] for item in iter_counts])
            out = f'   Mean number of iterations: {np.mean(niters):4.2f}'
            f.write(out + '\n')
            print(out)

            # filter statistics by type (error after time-step)
            PDE_errors = get_sorted(stats, type='PDE_error_after_step', sortby='time')
            coll_errors = get_sorted(stats, type='coll_error_after_step', sortby='time')
            for iters, PDE_err, coll_err in zip(iter_counts, PDE_errors, coll_errors):
                assert coll_err[1] < description['step_params']['errtol'], f'Error too high, got {coll_err[1]:8.4e}'
                out = (
                    f'   Errors after step {PDE_err[0]:8.4f} with {iters[1]} iterations: '
                    f'{PDE_err[1]:8.4e} / {coll_err[1]:8.4e}'
                )
                f.write(out + '\n')
                print(out)
            f.write('\n')
            print()

            # filter statistics by type (error after time-step)
            timing = get_sorted(stats, type='timing_run', sortby='time')
            out = f'...done, took {timing[0][1]} seconds!'
            f.write(out + '\n')
            print(out)

            print()
        out = '-----------------------------------------------------------------------------'
        f.write(out + '\n')
        print(out)

    f.close()
    assert np.isclose(
        mean_number_of_iterations, np.mean(niters), atol=1e-2
    ), f'Expected \
{mean_number_of_iterations:.2f} mean iterations, but got {np.mean(niters):.2f}'


def main():
    run_simulations(type='diffusion', ndim_list=[1], Tend=1.0, nsteps_list=[8], ml=False, nprocs=1)
    run_simulations(type='diffusion', ndim_list=[1], Tend=1.0, nsteps_list=[8], ml=True, nprocs=1)

    run_simulations(type='advection', ndim_list=[1], Tend=1.0, nsteps_list=[8], ml=False, nprocs=1)
    run_simulations(type='advection', ndim_list=[1], Tend=1.0, nsteps_list=[8], ml=True, nprocs=1)

    run_simulations(type='auzinger', ndim_list=[1], Tend=1.0, nsteps_list=[8], ml=False, nprocs=1)
    run_simulations(type='auzinger', ndim_list=[1], Tend=1.0, nsteps_list=[8], ml=True, nprocs=1)


if __name__ == "__main__":
    main()
