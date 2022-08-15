from pathlib import Path
import numpy as np

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.PenningTrap_3D import penningtrap
from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order
from pySDC.tutorial.step_3.HookClass_Particles import particle_hook


def main():
    """
    A simple test program to show th eenergy deviation for different quadrature nodes
    """
    stats_dict = run_simulation()

    ediff = dict()
    Path("data").mkdir(parents=True, exist_ok=True)
    f = open('data/step_3_C_out.txt', 'w')
    for cclass, stats in stats_dict.items():
        # filter and convert/sort statistics by etot and iterations
        energy = get_sorted(stats, type='etot', sortby='iter')
        # compare base and final energy
        base_energy = energy[0][1]
        final_energy = energy[-1][1]
        ediff[cclass] = abs(base_energy - final_energy)
        out = "Energy deviation for %s: %12.8e" % (cclass, ediff[cclass])
        f.write(out + '\n')
        print(out)
    f.close()

    # set expected differences and check
    ediff_expect = dict()
    ediff_expect['RADAU-RIGHT'] = 15
    ediff_expect['LOBATTO'] = 1e-05
    ediff_expect['GAUSS'] = 3e-05
    for k, v in ediff.items():
        assert v < ediff_expect[k], "ERROR: energy deviated too much, got %s" % ediff[k]


def run_simulation():
    """
    A simple test program to run IMEX SDC for a single time step
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-06
    level_params['dt'] = 1.0 / 16

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['num_nodes'] = 3

    # initialize problem parameters
    problem_params = dict()
    problem_params['omega_E'] = 4.9
    problem_params['omega_B'] = 25.0
    problem_params['u0'] = np.array([[10, 0, 0], [100, 0, 100], [1], [1]], dtype=object)
    problem_params['nparts'] = 1
    problem_params['sig'] = 0.1

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize controller parameters
    controller_params = dict()
    controller_params['hook_class'] = particle_hook  # specialized hook class for more statistics and output
    controller_params['logger_level'] = 30  # reduce verbosity of each run

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = penningtrap
    description['problem_params'] = problem_params
    description['sweeper_class'] = boris_2nd_order
    description['level_params'] = level_params
    description['step_params'] = step_params

    # assemble and loop over list of collocation classes
    quad_types = ['RADAU-RIGHT', 'GAUSS', 'LOBATTO']
    stats_dict = dict()
    for qtype in quad_types:
        sweeper_params['quad_type'] = qtype
        description['sweeper_params'] = sweeper_params

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

        # gather stats in dictionary, collocation classes being the keys
        stats_dict[qtype] = stats

    return stats_dict


if __name__ == "__main__":
    main()
