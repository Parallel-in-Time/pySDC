import numpy as np

from implementations.controller_classes.allinclusive_multigrid_nonMPI import allinclusive_multigrid_nonMPI

from implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from implementations.problem_classes.PenningTrap_3D import penningtrap
from implementations.datatype_classes.particles import particles, fields
from implementations.sweeper_classes.boris_2nd_order import boris_2nd_order

from examples.tutorial.step_3.B_HookClass import particles_output

from pySDC.Plugins.stats_helper import filter_stats, sort_stats, get_list_of_types

def main():

    err, stats = run_simulation()

    # filter statistics type (etot)
    filtered_stats = filter_stats(stats, type='etot')

    # sort and convert stats to list, sorted by iteration numbers (only pre- and after-step are present here)
    energy = sort_stats(filtered_stats, sortby='iter')

    # get base energy and show difference
    base_energy = energy[0][1]
    for item in energy:
        print('Total energy and deviation in iteration %2i: %12.10f -- %12.8e' %(item[0], item[1], abs(base_energy-item[1])))

    assert abs(base_energy - energy[-1][1]) < 15 , 'ERROR: energy deviated too much, got %s' %(base_energy-energy[-1][1])
    assert err < 5E-04, "ERROR: solution is not as exact as expected, got %s" %err

def run_simulation():
    """
    A simple test program to run IMEX SDC for a single time step
    """
    # initialize level parameters
    level_params = {}
    level_params['restol'] = 1E-08
    level_params['dt'] = 1/16

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3

    # initialize problem parameters
    problem_params = {}
    problem_params['omega_E'] = 4.9
    problem_params['omega_B'] = 25.0
    problem_params['u0'] = np.array([[10, 0, 0], [100, 0, 100], [1], [1]])
    problem_params['nparts'] = 100
    problem_params['sig'] = 0.1

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 20

    # initialize controller parameters
    controller_params = {}
    controller_params['hook_class'] = particles_output

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = penningtrap
    description['problem_params'] = problem_params
    description['dtype_u'] = particles
    description['dtype_f'] = fields
    description['sweeper_class'] = boris_2nd_order
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # instantiate the controller (no controller parameters used here)
    controller = allinclusive_multigrid_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0.0
    Tend = level_params['dt']

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_init()

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # compute error compared to know exact solution for one particle
    uex = P.u_exact(Tend)
    err = np.linalg.norm(uex.pos.values - uend.pos.values, np.inf) / np.linalg.norm(uex.pos.values, np.inf)

    return err, stats


if __name__ == "__main__":
    main()
