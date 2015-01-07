from pySDC import CollocationClasses as collclass

import numpy as np
import matplotlib.pyplot as plt

from examples.penningtrap.ProblemClass import penningtrap
from examples.penningtrap.TransferClass import particles_to_particles
from pySDC.datatype_classes.particles import particles, fields
from pySDC.sweeper_classes.boris_2nd_order import boris_2nd_order
from examples.penningtrap.HookClass import particles_output
import pySDC.Methods_Parallel as mp
from pySDC import Log
from pySDC.Stats import grep_stats, sort_stats


if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 1

    # This comes as read-in for each level
    lparams = {}
    lparams['restol'] = 5E-12

    # This comes as read-in for the time-stepping
    sparams = {}
    sparams['maxiter'] = 10

    # This comes as read-in for the problem
    pparams = {}
    pparams['omega_E'] = 4.9
    pparams['omega_B'] = 25.0
    pparams['u0'] = np.array([[10,0,0],[100,0,100],[1],[1]])
    pparams['nparts'] = 10
    pparams['sig'] = 0.1

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = penningtrap
    description['problem_params'] = pparams
    description['dtype_u'] = particles
    description['dtype_f'] = fields
    description['collocation_class'] = collclass.CollGaussLobatto
    description['num_nodes'] = [3]
    description['sweeper_class'] = boris_2nd_order
    description['level_params'] = lparams
    description['transfer_class'] = particles_to_particles # this is only needed for more than 2 levels
    description['hook_class'] = particles_output # this is optional

    # quickly generate block of steps
    MS = mp.generate_steps(num_procs,sparams,description)

    # setup parameters "in time"
    t0 = 0
    dt = 0.015625
    Tend = 2*dt

    # get initial values on finest level
    P = MS[0].levels[0].prob
    uinit = P.u_init()

    # call main function to get things done...
    uend,stats = mp.run_pfasst_serial(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

    extract_stats = grep_stats(stats,type='etot')
    sortedlist_stats = sort_stats(extract_stats,sortby='time')
    print(extract_stats,sortedlist_stats)

    plt.show()
