import numpy as np
from pathlib import Path

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.PenningTrap_3D import penningtrap
from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order
from pySDC.tutorial.step_3.HookClass_Particles import particle_hook


def main():
    """
    A simple tets program to retrieve user-defined statistics from a run
    """
    Path("data").mkdir(parents=True, exist_ok=True)

    err, stats = run_penning_trap_simulation()

    # filter statistics type (etot)
    filtered_stats = filter_stats(stats, type='etot')

    # sort and convert stats to list, sorted by iteration numbers (only pre- and after-step are present here)
    energy = sort_stats(filtered_stats, sortby='iter')

    # get base energy and show difference
    base_energy = energy[0][1]
    f = open('data/step_3_B_out.txt', 'a')
    for item in energy:
        out = 'Total energy and deviation in iteration %2i: %12.10f -- %12.8e' % (
            item[0],
            item[1],
            abs(base_energy - item[1]),
        )
        f.write(out + '\n')
        print(out)
    f.close()

    assert abs(base_energy - energy[-1][1]) < 15, 'ERROR: energy deviated too much, got %s' % (
        base_energy - energy[-1][1]
    )
    assert err < 5e-04, "ERROR: solution is not as exact as expected, got %s" % err


def run_penning_trap_simulation():
    """
    A simple test program to run IMEX SDC for a single time step of the penning trap example
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-08
    level_params['dt'] = 1.0 / 16

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3

    # initialize problem parameters for the Penning trap
    problem_params = dict()
    problem_params['omega_E'] = 4.9  # E-field frequency
    problem_params['omega_B'] = 25.0  # B-field frequency
    problem_params['u0'] = np.array([[10, 0, 0], [100, 0, 100], [1], [1]], dtype=object)  # initial center of positions
    problem_params['nparts'] = 1  # number of particles in the trap
    problem_params['sig'] = 0.1  # smoothing parameter for the forces

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize controller parameters
    controller_params = dict()
    controller_params['hook_class'] = particle_hook  # specialized hook class for more statistics and output
    controller_params['log_to_file'] = True
    controller_params['fname'] = 'data/step_3_B_out.txt'

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = penningtrap
    description['problem_params'] = problem_params
    description['sweeper_class'] = boris_2nd_order
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # instantiate the controller (no controller parameters used here)
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

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
    err = np.linalg.norm(uex.pos - uend.pos, np.inf) / np.linalg.norm(uex.pos, np.inf)

    return err, stats


if __name__ == "__main__":
    main()
