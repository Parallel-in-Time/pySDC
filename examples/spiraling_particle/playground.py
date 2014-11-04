from pySDC import Step as stepclass
from pySDC import Level as levclass
from pySDC import CollocationClasses as collclass

import numpy as np
import matplotlib.pyplot as plt

from examples.spiraling_particle.ProblemClass import planewave_single
from pySDC.datatype_classes.particles import particles, fields
from pySDC.sweeper_classes.boris_2nd_order import boris_2nd_order
from pySDC.Methods import sdc_step


if __name__ == "__main__":

    # This comes as read-in for each level
    lparams = {}
    lparams['restol'] = 1E-08

    # This comes as read-in for the time-stepping
    sparams = {}
    sparams['Tend'] = np.pi/10
    sparams['maxiter'] = 10

    # This comes as read-in for the problem
    cparams_f = {}
    cparams_f['delta'] = 1
    cparams_f['a0'] = 0.01
    cparams_f['u0'] = np.array([[0,-1,0],[0.05,0.01,0]])
    cparams_f['alpha'] = 1

    # definition of the fine level (which is the only one we care about here)
    L0 = levclass.level(problem_class       =   planewave_single,
                        problem_params      =   cparams_f,
                        dtype_u             =   particles,
                        dtype_f             =   fields,
                        collocation_class   =   collclass.CollGaussLobatto,
                        num_nodes           =   2,
                        sweeper_class       =   boris_2nd_order,
                        level_params        =   lparams,
                        id                  =   'L0')

    # create a time step object and add the level to it
    S = stepclass.step(sparams)
    S.register_level(L0)

    # set some initial parameters
    S.time = 0
    S.dt = np.pi/10
    S.stats.niter = 0

    # compute initial values
    P = S.levels[0].prob
    uinit = P.u_exact(S.time)

    # initialize the step
    S.init_step(uinit)
    print('Init:',S.levels[0].u[0].pos.values,S.levels[0].u[0].vel.values)

    fig = plt.figure(figsize=(10,10))
    plt.ion()
    plt.axis([-1.5, 1.5, -1.5, 1.5])

    hl, = plt.plot(S.levels[0].u[0].pos.values[0],S.levels[0].u[0].pos.values[1],'b-')

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

        hl.set_xdata(np.append(hl.get_xdata(),uend.pos.values[0]))
        hl.set_ydata(np.append(hl.get_ydata(),uend.pos.values[1]))
        plt.draw()

        # reset step (not necessary, but cleaner)
        S.reset_step()

        # initialize next step
        S.init_step(uend)

    # plt.savefig('2300_002.pdf')
    plt.show()

    # get exact values and print the error
    uex = P.u_exact(S.params.Tend)
    print(uex.pos.values,uend.pos.values)
    print('Error:',np.linalg.norm(uex.pos.values-uend.pos.values,np.inf)/np.linalg.norm(uex.pos.values,np.inf))