import numpy as np

from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced import fenics_heat_mass, fenics_heat
from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_weak_forced import fenics_heat_weak_imex
from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh, rhs_fenics_mesh
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass, imex_1st_order
from pySDC.implementations.transfer_classes.BaseTransfer_mass import base_transfer_mass

from pySDC.helpers.stats_helper import filter_stats, sort_stats


def setup(t0=None):
    """
    Helper routine to set up parameters

    Args:
        t0 (float): initial time

    Returns:
        description and controller_params parameter dictionaries
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 5E-10
    level_params['dt'] = 0.2

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]

    problem_params = dict()
    problem_params['nu'] = 0.1
    problem_params['t0'] = t0  # ugly, but necessary to set up this ProblemClass
    problem_params['c_nvars'] = [128]
    problem_params['family'] = 'CG'
    problem_params['order'] = [4]
    problem_params['refinements'] = [1]

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30

    base_transfer_params = dict()
    # base_transfer_params['finter'] = True

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = None
    description['problem_params'] = problem_params
    description['dtype_u'] = fenics_mesh
    description['dtype_f'] = rhs_fenics_mesh
    description['sweeper_class'] = None
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['base_transfer_params'] = base_transfer_params

    return description, controller_params


def run_sdc_variants(variant=None):
    """
    Main routine to run the different implementations of the heat equation with FEniCS

    Args:
        variant (str): specifies the variant
    """
    Tend = 1.0
    num_procs = 1
    t0 = 0.0

    description, controller_params = setup(t0=t0)

    if variant == 'mass':
        # Note that we need to reduce the tolerance for the residual here, since otherwise the error will be too high
        description['level_params']['restol'] /= 500
        description['problem_class'] = fenics_heat_mass
        description['sweeper_class'] = imex_1st_order_mass
        description['base_transfer_class'] = base_transfer_mass
    elif variant == 'mass_inv':
        description['problem_class'] = fenics_heat
        description['sweeper_class'] = imex_1st_order
    elif variant == 'weak':
        description['problem_class'] = fenics_heat_weak_imex
        description['sweeper_class'] = imex_1st_order
    else:
        raise NotImplementedError('Variant %s is not implemented' % variant)

    # quickly generate block of steps
    controller = controller_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(0.0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # compute exact solution and compare
    uex = P.u_exact(Tend)
    err = abs(uex - uend) / abs(uex)

    out = 'Variant %s -- error at time %s: %s' % (variant, Tend, err)
    print(out)

    # filter statistics by type (number of iterations)
    filtered_stats = filter_stats(stats, type='niter')

    # convert filtered statistics to list of iterations count, sorted by process
    iter_counts = sort_stats(filtered_stats, sortby='time')

    niters = np.array([item[1] for item in iter_counts])
    out = '   Mean number of iterations: %4.2f' % np.mean(niters)
    print(out)
    out = '   Range of values for number of iterations: %2i ' % np.ptp(niters)
    print(out)
    out = '   Position of max/min number of iterations: %2i -- %2i' % \
          (int(np.argmax(niters)), int(np.argmin(niters)))
    print(out)
    out = '   Std and var for number of iterations: %4.2f -- %4.2f' % (float(np.std(niters)), float(np.var(niters)))
    print(out)

    assert np.mean(niters) <= 6.0, 'Mean number of iterations is too high, got %s' % np.mean(niters)
    assert err <= 4.04E-08, 'Error is too high, got %s' % err

    print()


def main():
    run_sdc_variants(variant='mass_inv')
    run_sdc_variants(variant='mass')
    run_sdc_variants(variant='weak')


if __name__ == "__main__":
    main()
