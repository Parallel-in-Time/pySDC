from pathlib import Path
import numpy as np

from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced import (
    fenics_heat_mass,
    fenics_heat,
    fenics_heat_mass_timebc,
)
from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass, imex_1st_order
from pySDC.implementations.transfer_classes.TransferFenicsMesh import mesh_to_mesh_fenics


def setup(t0=None, ml=None):
    """
    Helper routine to set up parameters

    Args:
        t0 (float): initial time
        ml (bool): use single or multiple levels

    Returns:
        description and controller_params parameter dictionaries
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 5e-10
    level_params['dt'] = 0.2

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    if ml:
        # Note that coarsening in the nodes actually HELPS MLSDC to converge (M=1 is exact on the coarse level)
        sweeper_params['num_nodes'] = [3, 1]
    else:
        sweeper_params['num_nodes'] = [3]

    problem_params = dict()
    problem_params['nu'] = 0.1
    problem_params['t0'] = t0  # ugly, but necessary to set up this ProblemClass
    problem_params['c_nvars'] = [128]
    problem_params['family'] = 'CG'
    problem_params['c'] = 1.0
    if ml:
        # We can do rather aggressive coarsening here. As long as we have 1 node on the coarse level, all is "well" (ie.
        # MLSDC does not take more iterations than SDC, but also not less). If we just coarsen in the refinement (and
        # not in the nodes and order, the mass inverse approach is way better, ie. halves the number of iterations!
        problem_params['order'] = [4, 1]
        problem_params['refinements'] = [1, 0]
    else:
        problem_params['order'] = [4]
        problem_params['refinements'] = [1]

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30

    base_transfer_params = dict()
    base_transfer_params['finter'] = True

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = None
    description['problem_params'] = problem_params
    description['sweeper_class'] = None
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['space_transfer_class'] = mesh_to_mesh_fenics
    description['base_transfer_params'] = base_transfer_params

    return description, controller_params


def run_variants(variant=None, ml=None, num_procs=None):
    """
    Main routine to run the different implementations of the heat equation with FEniCS

    Args:
        variant (str): specifies the variant
        ml (bool): use single or multiple levels
        num_procs (int): number of processors in time
    """
    Tend = 1.0
    t0 = 0.0

    description, controller_params = setup(t0=t0, ml=ml)

    if variant == 'mass':
        # Note that we need to reduce the tolerance for the residual here, since otherwise the error will be too high
        description['level_params']['restol'] /= 500
        description['problem_class'] = fenics_heat_mass
        description['sweeper_class'] = imex_1st_order_mass
    elif variant == 'mass_inv':
        description['problem_class'] = fenics_heat
        description['sweeper_class'] = imex_1st_order
    elif variant == 'mass_timebc':
        # Can increase the tolerance here, errors are higher anyway
        description['level_params']['restol'] *= 20
        description['problem_class'] = fenics_heat_mass_timebc
        description['sweeper_class'] = imex_1st_order_mass
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

    Path("data").mkdir(parents=True, exist_ok=True)
    f = open('data/step_7_A_out.txt', 'a')

    out = f'Variant {variant} with ml={ml} and num_procs={num_procs} -- error at time {Tend}: {err}'
    f.write(out + '\n')
    print(out)

    # filter statistics by type (number of iterations)
    iter_counts = get_sorted(stats, type='niter', sortby='time')

    niters = np.array([item[1] for item in iter_counts])
    out = '   Mean number of iterations: %4.2f' % np.mean(niters)
    f.write(out + '\n')
    print(out)
    out = '   Range of values for number of iterations: %2i ' % np.ptp(niters)
    f.write(out + '\n')
    print(out)
    out = '   Position of max/min number of iterations: %2i -- %2i' % (int(np.argmax(niters)), int(np.argmin(niters)))
    f.write(out + '\n')
    print(out)
    out = '   Std and var for number of iterations: %4.2f -- %4.2f' % (float(np.std(niters)), float(np.var(niters)))
    f.write(out + '\n')
    print(out)

    timing = get_sorted(stats, type='timing_run', sortby='time')
    out = f'Time to solution: {timing[0][1]:6.4f} sec.'
    f.write(out + '\n')
    print(out)

    if num_procs == 1:
        assert np.mean(niters) <= 6.0, 'Mean number of iterations is too high, got %s' % np.mean(niters)
        if variant == 'mass' or variant == 'mass_inv':
            assert err <= 1.14e-08, 'Error is too high, got %s' % err
        else:
            assert err <= 3.25e-07, 'Error is too high, got %s' % err
    else:
        assert np.mean(niters) <= 11.6, 'Mean number of iterations is too high, got %s' % np.mean(niters)
        assert err <= 1.14e-08, 'Error is too high, got %s' % err

    f.write('\n')
    print()
    f.close()


def main():
    run_variants(variant='mass_inv', ml=False, num_procs=1)
    run_variants(variant='mass', ml=False, num_procs=1)
    run_variants(variant='mass_timebc', ml=False, num_procs=1)
    run_variants(variant='mass_inv', ml=True, num_procs=1)
    run_variants(variant='mass', ml=True, num_procs=1)
    run_variants(variant='mass_timebc', ml=True, num_procs=1)
    run_variants(variant='mass_inv', ml=True, num_procs=5)

    # WARNING: all other variants do NOT work, either because of FEniCS restrictions (weak forms with different meshes
    # will not work together) or because of inconsistent use of the mass matrix (locality condition for the tau
    # correction is not satisfied, mass matrix does not permute with restriction).
    # run_pfasst_variants(variant='mass', ml=True, num_procs=5)


if __name__ == "__main__":
    main()
