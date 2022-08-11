import numpy as np

from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced import fenics_heat_mass
from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
from pySDC.implementations.transfer_classes.BaseTransfer_mass import base_transfer_mass
from pySDC.implementations.transfer_classes.TransferFenicsMesh import mesh_to_mesh_fenics
from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced import fenics_heat
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order

from pySDC.playgrounds.FEniCS.HookClass_FEniCS_output import fenics_output

from pySDC.helpers.stats_helper import get_sorted
import pySDC.helpers.plot_helper as plt_helper


def run_simulation(ml=None, mass=None):

    t0 = 0
    dt = 0.2
    Tend = 0.2

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-10
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
    problem_params['order'] = [4]
    if ml:
        problem_params['refinements'] = [1, 0]
    else:
        problem_params['refinements'] = [1]

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = fenics_output

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    if mass:
        description['problem_class'] = fenics_heat_mass
        description['sweeper_class'] = imex_1st_order_mass
        description['base_transfer_class'] = base_transfer_mass
    else:
        description['problem_class'] = fenics_heat
        description['sweeper_class'] = imex_1st_order  # pass sweeper (see part B)
    description['problem_params'] = problem_params
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['space_transfer_class'] = mesh_to_mesh_fenics

    # quickly generate block of steps
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    errors = get_sorted(stats, type='error', sortby='iter')
    residuals = get_sorted(stats, type='residual', sortby='iter')

    return errors, residuals


def visualize():

    errors_sdc_M = np.load('errors_sdc_M.npy')
    errors_sdc_noM = np.load('errors_sdc_noM.npy')
    errors_mlsdc_M = np.load('errors_mlsdc_M.npy')
    errors_mlsdc_noM = np.load('errors_mlsdc_noM.npy')

    plt_helper.setup_mpl()

    plt_helper.newfig(240, 1, ratio=0.8)

    plt_helper.plt.semilogy(
        [err[0] for err in errors_sdc_noM],
        [err[1] for err in errors_sdc_noM],
        lw=2,
        marker='s',
        markersize=6,
        color='darkblue',
        label='SDC without M',
    )

    plt_helper.plt.xlim([0, 11])
    plt_helper.plt.ylim([6e-09, 2e-03])
    plt_helper.plt.xlabel('iteration')
    plt_helper.plt.ylabel('error')
    plt_helper.plt.legend()
    plt_helper.plt.grid()

    plt_helper.savefig('error_SDC_noM_CG_4')

    plt_helper.newfig(240, 1, ratio=0.8)

    plt_helper.plt.semilogy(
        [err[0] for err in errors_sdc_noM],
        [err[1] for err in errors_sdc_noM],
        lw=2,
        color='darkblue',
        marker='s',
        markersize=6,
        label='SDC without M',
    )
    plt_helper.plt.semilogy(
        [err[0] for err in errors_sdc_M],
        [err[1] for err in errors_sdc_M],
        lw=2,
        marker='o',
        markersize=6,
        color='red',
        label='SDC with M',
    )

    plt_helper.plt.xlim([0, 11])
    plt_helper.plt.ylim([6e-09, 2e-03])
    plt_helper.plt.xlabel('iteration')
    plt_helper.plt.ylabel('error')
    plt_helper.plt.legend()
    plt_helper.plt.grid()

    plt_helper.savefig('error_SDC_M_CG_4')

    plt_helper.newfig(240, 1, ratio=0.8)

    plt_helper.plt.semilogy(
        [err[0] for err in errors_mlsdc_noM],
        [err[1] for err in errors_mlsdc_noM],
        lw=2,
        marker='s',
        markersize=6,
        color='darkblue',
        label='MLSDC without M',
    )

    plt_helper.plt.xlim([0, 11])
    plt_helper.plt.ylim([6e-09, 2e-03])
    plt_helper.plt.xlabel('iteration')
    plt_helper.plt.ylabel('error')
    plt_helper.plt.legend()
    plt_helper.plt.grid()

    plt_helper.savefig('error_MLSDC_noM_CG_4')

    plt_helper.newfig(240, 1, ratio=0.8)

    plt_helper.plt.semilogy(
        [err[0] for err in errors_mlsdc_noM],
        [err[1] for err in errors_mlsdc_noM],
        lw=2,
        color='darkblue',
        marker='s',
        markersize=6,
        label='MLSDC without M',
    )
    plt_helper.plt.semilogy(
        [err[0] for err in errors_mlsdc_M],
        [err[1] for err in errors_mlsdc_M],
        lw=2,
        marker='o',
        markersize=6,
        color='red',
        label='MLSDC with M',
    )

    plt_helper.plt.xlim([0, 11])
    plt_helper.plt.ylim([6e-09, 2e-03])
    plt_helper.plt.xlabel('iteration')
    plt_helper.plt.ylabel('error')
    plt_helper.plt.legend()
    plt_helper.plt.grid()

    plt_helper.savefig('error_MLSDC_M_CG_4')


if __name__ == "__main__":

    # errors_sdc_noM, _ = run_simulation(ml=False, mass=False)
    # errors_sdc_M, _ = run_simulation(ml=False, mass=True)
    # errors_mlsdc_noM, _ = run_simulation(ml=True, mass=False)
    # errors_mlsdc_M, _ = run_simulation(ml=True, mass=True)
    #
    # np.save('errors_sdc_M.npy',  errors_sdc_M)
    # np.save('errors_sdc_noM.npy',  errors_sdc_noM)
    # np.save('errors_mlsdc_M.npy',  errors_mlsdc_M)
    # np.save('errors_mlsdc_noM.npy',  errors_mlsdc_noM)

    visualize()
