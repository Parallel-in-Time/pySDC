from __future__ import division
from subprocess import call

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

from pySDC import CollocationClasses as collclass
from examples.spiraling_particle.ProblemClass import planewave_single
from pySDC.datatype_classes.particles import particles, fields
from pySDC.sweeper_classes.boris_2nd_order import boris_2nd_order
from examples.spiraling_particle.HookClass import particles_output
import pySDC.Methods as mp
from pySDC import Log
from pySDC.Stats import grep_stats, sort_stats


if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 1

    # This comes as read-in for each level
    lparams = {}
    lparams['restol'] = 1E-12

    # This comes as read-in for the time-stepping
    sparams = {}
    sparams['maxiter'] = 20

    # This comes as read-in for the problem
    pparams = {}
    pparams['delta'] = 1
    pparams['a0'] = 0.01
    pparams['u0'] = np.array([[0,-1,0],[0.05,0.01,0],[1],[1]])

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = planewave_single
    description['problem_params'] = pparams
    description['dtype_u'] = particles
    description['dtype_f'] = fields
    description['collocation_class'] = collclass.CollGaussLobatto
    description['num_nodes'] = [3]
    description['sweeper_class'] = boris_2nd_order
    description['level_params'] = lparams
    # description['transfer_class'] = particles_to_particles # this is only needed for more than 2 levels
    description['hook_class'] = particles_output # this is optional

    # quickly generate block of steps
    MS = mp.generate_steps(num_procs,sparams,description)

    # setup parameters "in time"
    t0 = 0
    dt = 1/16*np.pi
    Tend = 16*dt#np.pi

    # get initial values on finest level
    P = MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    print('Init:',uinit.pos.values,uinit.vel.values)

    # call main function to get things done...
    uend,stats = mp.run_pfasst_serial(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

    extract_stats = grep_stats(stats,type='energy')
    sortedlist_stats = sort_stats(extract_stats,sortby='time')

    R0 = np.linalg.norm(uinit.pos.values[:])
    H0 = 1/2*np.dot(uinit.vel.values[:],uinit.vel.values[:])+0.02/R0

    energy_err = [abs(entry[1]-H0)/H0 for entry in sortedlist_stats]

    fig = plt.figure()
    plt.plot(energy_err,'bo--')

    plt.show()

# rc('text', usetex=True)
# rc('font', family='serif', size=20)
# rc('legend', fontsize='small')
# rc('xtick', labelsize='small')
# rc('ytick', labelsize='small')
#
#
# guide1 = []
# guide2 = []
#
# for n in range(len(dt_list)):
#     guide2.append(energy_err[0]*1.0/(dt_list[0]/dt_list[n])**(2*M-2))
#     guide1.append(energy_err[0]*1.0/(dt_list[0]/dt_list[n])**2)
#
# fig = plt.figure(figsize=(14,8))
#
# plt.loglog(dt_list, energy_err, label='Boris-SDC, M=3, tol=1E-12' ,color='r', linestyle='-', linewidth=2, marker='o',
#            markersize=10, markerfacecolor='r', markeredgewidth=3, markeredgecolor='k')
# plt.loglog(dt_list, guide1, label='order 2', color='k', linestyle='--', linewidth=2)
# plt.loglog(dt_list, guide2, label='order 4', color='b', linestyle='-.', linewidth=2)
#
# plt.xlabel('$\Delta t/\pi$')
# plt.ylabel('$H(T)/H(0)$')
# plt.legend(numpoints=1, shadow=True,loc=0)
# plt.xticks(dt_list,dt_list/np.pi)
# plt.axis([1.1*dt_list[0],0.9*dt_list[-1],1E-14,1E-02])
# # plt.xticks(x,['256', '512', '1K', '2K', '4K', '8K', '16K', '32K', '64K', '112K', '224K', '448K'])
# # plt.yticks(y_ideal,y_ideal)
# plt.grid('on')
# plt.tight_layout()
#
# name = 'SPIRAL_energy_error_M3_tol1E-12.pdf'
# plt.savefig(name,rasterized=True)
# call('pdfcrop '+name+' '+name,shell=True)
#
# plt.show()