import time

import numpy as np
from pySDC.tutorial.step_4.PenningTrap_3D_coarse import penningtrap_coarse
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI
from pySDC.implementations.problem_classes.PenningTrap_3D import penningtrap
from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order
from pySDC.implementations.transfer_classes.TransferParticles_NoCoarse import particles_to_particles

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.datatype_classes.particles import particles, fields
from pySDC.tutorial.step_3.HookClass_Particles import particle_hook


def main():
    """
    A simple test program to compare SDC with two flavors of MLSDC for particle dynamics
    """

    # run SDC, MLSDC and MLSDC plus f-interpolation and compare
    stats_sdc, time_sdc = run_penning_trap_simulation(mlsdc=False)
    stats_mlsdc, time_mlsdc = run_penning_trap_simulation(mlsdc=True)
    stats_mlsdc_finter, time_mlsdc_finter = run_penning_trap_simulation(mlsdc=True, finter=True)

    f = open('step_4_D_out.txt', 'w')
    out = 'Timings for SDC, MLSDC and MLSDC+finter: %12.8f -- %12.8f -- %12.8f' % \
          (time_sdc, time_mlsdc, time_mlsdc_finter)
    f.write(out + '\n')
    print(out)

    # filter statistics type (etot)
    filtered_stats_sdc = filter_stats(stats_sdc, type='etot')
    filtered_stats_mlsdc = filter_stats(stats_mlsdc, type='etot')
    filtered_stats_mlsdc_finter = filter_stats(stats_mlsdc_finter, type='etot')

    # sort and convert stats to list, sorted by iteration numbers (only pre- and after-step are present here)
    energy_sdc = sort_stats(filtered_stats_sdc, sortby='iter')
    energy_mlsdc = sort_stats(filtered_stats_mlsdc, sortby='iter')
    energy_mlsdc_finter = sort_stats(filtered_stats_mlsdc_finter, sortby='iter')

    # get base energy and show differences
    base_energy = energy_sdc[0][1]
    for item in energy_sdc:
        out = 'Total energy and relative deviation in iteration %2i: %12.10f -- %12.8e' % \
              (item[0], item[1], abs(base_energy - item[1]) / base_energy)
        f.write(out + '\n')
        print(out)
    for item in energy_mlsdc:
        out = 'Total energy and relative deviation in iteration %2i: %12.10f -- %12.8e' % \
              (item[0], item[1], abs(base_energy - item[1]) / base_energy)
        f.write(out + '\n')
        print(out)
    for item in energy_mlsdc_finter:
        out = 'Total energy and relative deviation in iteration %2i: %12.10f -- %12.8e' % \
              (item[0], item[1], abs(base_energy - item[1]) / base_energy)
        f.write(out + '\n')
        print(out)
    f.close()

    assert abs(energy_sdc[-1][1] - energy_mlsdc[-1][1]) / base_energy < 6E-10, \
        'ERROR: energy deviated too much between SDC and MLSDC, got %s' % (
        abs(energy_sdc[-1][1] - energy_mlsdc[-1][1]) / base_energy)
    assert abs(energy_mlsdc[-1][1] - energy_mlsdc_finter[-1][1]) / base_energy < 8E-10, \
        'ERROR: energy deviated too much after using finter, got %s' % (
        abs(energy_mlsdc[-1][1] - energy_mlsdc_finter[-1][1]) / base_energy)


def run_penning_trap_simulation(mlsdc, finter=False):
    """
    A simple test program to run IMEX SDC for a single time step
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-07
    level_params['dt'] = 1.0 / 8

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 5

    # initialize problem parameters for the Penning trap
    problem_params = dict()
    problem_params['omega_E'] = 4.9  # E-field frequency
    problem_params['omega_B'] = 25.0  # B-field frequency
    problem_params['u0'] = np.array([[10, 0, 0], [100, 0, 100], [1], [1]])  # initial center of positions
    problem_params['nparts'] = 50  # number of particles in the trap
    problem_params['sig'] = 0.1  # smoothing parameter for the forces

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize controller parameters
    controller_params = dict()
    controller_params['hook_class'] = particle_hook  # specialized hook class for more statistics and output
    controller_params['logger_level'] = 30

    transfer_params = dict()
    transfer_params['finter'] = finter

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    if mlsdc:
        # MLSDC: provide list of two problem classes: one for the fine, one for the coarse level
        description['problem_class'] = [penningtrap, penningtrap_coarse]
    else:
        # SDC: provide only one problem class
        description['problem_class'] = penningtrap
    description['problem_params'] = problem_params
    description['dtype_u'] = particles
    description['dtype_f'] = fields
    description['sweeper_class'] = boris_2nd_order
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['space_transfer_class'] = particles_to_particles
    description['base_transfer_params'] = transfer_params

    # instantiate the controller (no controller parameters used here)
    controller = allinclusive_classic_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0.0
    Tend = level_params['dt']

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_init()

    # call and time main function to get things done...
    start_time = time.time()
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    end_time = time.time() - start_time

    return stats, end_time


if __name__ == "__main__":
    main()
