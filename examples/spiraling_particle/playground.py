from pySDC import Step as stepclass
from pySDC import Level as levclass
from pySDC import CollocationClasses as collclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from subprocess import call

from examples.spiraling_particle.ProblemClass import planewave_single
from pySDC.datatype_classes.particles import particles, fields
from pySDC.sweeper_classes.boris_2nd_order import boris_2nd_order
from pySDC.Methods import sdc_step


if __name__ == "__main__":

    # This comes as read-in for each level
    lparams = {}
    lparams['restol'] = 1E-12

    # This comes as read-in for the time-stepping
    sparams = {}
    sparams['Tend'] = np.pi
    sparams['maxiter'] = 20

    # This comes as read-in for the problem
    cparams_f = {}
    cparams_f['delta'] = 1
    cparams_f['a0'] = 0.01
    cparams_f['u0'] = np.array([[0,-1,0],[0.05,0.01,0]])
    cparams_f['alpha'] = 1

    M = 3

    # definition of the fine level (which is the only one we care about here)
    L0 = levclass.level(problem_class       =   planewave_single,
                        problem_params      =   cparams_f,
                        dtype_u             =   particles,
                        dtype_f             =   fields,
                        collocation_class   =   collclass.CollGaussLobatto,
                        num_nodes           =   M,
                        sweeper_class       =   boris_2nd_order,
                        level_params        =   lparams,
                        id                  =   'L0')

    # create a time step object and add the level to it
    S = stepclass.step(sparams)
    S.register_level(L0)


    dt_list = np.array([1/4,1/8,1/16,1/32,1/64,1/128])*np.pi
    energy_err = []
    for dt in dt_list:

        # set some initial parameters
        S.time = 0
        S.dt = dt
        S.stats.niter = 0

        # compute initial values
        P = S.levels[0].prob
        uinit = P.u_exact(S.time)

        # initialize the step
        S.init_step(uinit)
        print('Init:',S.levels[0].u[0].pos.values,S.levels[0].u[0].vel.values)

        # fig = plt.figure(figsize=(10,10))
        # plt.ion()
        # plt.axis([-1.5, 1.5, -1.5, 1.5])

        # hl, = plt.plot(S.levels[0].u[0].pos.values[0],S.levels[0].u[0].pos.values[1],'b-')

        # do the time-stepping
        step_stats = []
        nsteps = int(S.params.Tend/S.dt)

        for n in range(nsteps):

            # call SDC for the current step
            uend = sdc_step(S)

            # save stats separated from the stepping object
            step_stats.append(S.stats)

            # advance time
            S.time += S.dt

            # hl.set_xdata(np.append(hl.get_xdata(),uend.pos.values[0]))
            # hl.set_ydata(np.append(hl.get_ydata(),uend.pos.values[1]))
            # plt.draw()

            # reset step (not necessary, but cleaner)
            S.reset_step()

            # initialize next step
            S.init_step(uend)

        energy_err.append(step_stats[-1].level_stats[0].energy_err)
        print('Energy Error at time %s for step size %s: %s' %(S.time,dt,energy_err[-1]))
        # plt.savefig('2300_002.pdf')
        # plt.show()

        # get exact values and print the error
        # uex = P.u_exact(S.params.Tend)
        # print(uex.pos.values,uend.pos.values)
        # print('Error:',np.linalg.norm(uex.pos.values-uend.pos.values,np.inf)/np.linalg.norm(uex.pos.values,np.inf))


rc('text', usetex=True)
rc('font', family='serif', size=20)
rc('legend', fontsize='small')
rc('xtick', labelsize='small')
rc('ytick', labelsize='small')


guide1 = []
guide2 = []

for n in range(len(dt_list)):
    guide2.append(energy_err[0]*1.0/(dt_list[0]/dt_list[n])**(2*M-2))
    guide1.append(energy_err[0]*1.0/(dt_list[0]/dt_list[n])**2)

fig = plt.figure(figsize=(14,8))

plt.loglog(dt_list, energy_err, label='Boris-SDC, M=3, tol=1E-12' ,color='r', linestyle='-', linewidth=2, marker='o',
           markersize=10, markerfacecolor='r', markeredgewidth=3, markeredgecolor='k')
plt.loglog(dt_list, guide1, label='order 2', color='k', linestyle='--', linewidth=2)
plt.loglog(dt_list, guide2, label='order 4', color='b', linestyle='-.', linewidth=2)

plt.xlabel('$\Delta t/\pi$')
plt.ylabel('$H(T)/H(0)$')
plt.legend(numpoints=1, shadow=True,loc=0)
plt.xticks(dt_list,dt_list/np.pi)
plt.axis([1.1*dt_list[0],0.9*dt_list[-1],1E-14,1E-02])
# plt.xticks(x,['256', '512', '1K', '2K', '4K', '8K', '16K', '32K', '64K', '112K', '224K', '448K'])
# plt.yticks(y_ideal,y_ideal)
plt.grid('on')
plt.tight_layout()

name = 'SPIRAL_energy_error_M3_tol1E-12.pdf'
plt.savefig(name,rasterized=True)
call('pdfcrop '+name+' '+name,shell=True)

plt.show()