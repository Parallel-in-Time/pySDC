from pySDC import Step as stepclass
from pySDC import Level as levclass
from pySDC import CollocationClasses as collclass

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from examples.penningtrap.ProblemClass import penningtrap
from examples.penningtrap.TransferClass import particles_to_particles
from pySDC.datatype_classes.particles import particles, fields
from pySDC.sweeper_classes.boris_2nd_order import boris_2nd_order
from pySDC.Methods import sdc_step, mlsdc_step


if __name__ == "__main__":

    # This comes as read-in for each level
    lparams = {}
    lparams['restol'] = 1E-12

    # This comes as read-in for the time-stepping
    sparams = {}
    sparams['Tend'] = 500*0.015625
    sparams['maxiter'] = 10

    # This comes as read-in for the problem
    cparams_f = {}
    cparams_f['omega_E'] = 4.9
    cparams_f['omega_B'] = 25.0
    cparams_f['alpha'] = 1
    cparams_f['u0'] = np.array([[10,0,0],[100,0,100]])
    cparams_f['eps'] = -1
    cparams_f['nparts'] = 10
    cparams_f['sig'] = 0.1

    # definition of the fine level (which is the only one we care about here)
    L0 = levclass.level(problem_class       =   penningtrap,
                        problem_params      =   cparams_f,
                        dtype_u             =   particles,
                        dtype_f             =   fields,
                        collocation_class   =   collclass.CollGaussLegendre,
                        num_nodes           =   3,
                        sweeper_class       =   boris_2nd_order,
                        level_params        =   lparams,
                        id                  =   'L0')

    # create a time step object and add the level to it
    S = stepclass.step(sparams)
    S.register_level(L0)

    # set some initial parameters
    S.time = 0
    S.dt = 0.015625
    S.stats.niter = 0

    # compute initial values
    P = S.levels[0].prob
    uinit = P.u_init()

    # initialize the step
    S.init_step(uinit)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d([-20,20])
    ax.set_ylim3d([-20,20])
    ax.set_zlim3d([-20,20])
    plt.ion()
    sframe = None

    # do the time-stepping
    step_stats = []
    nsteps = int(S.params.Tend/S.dt)
    for n in range(nsteps):

        # call SDC for the current step
        uend = sdc_step(S)

        oldcol = sframe
        sframe = ax.scatter(uend.pos.values[0::3],uend.pos.values[1::3],uend.pos.values[2::3])
        # Remove old line collection before drawing
        if oldcol is not None:
            ax.collections.remove(oldcol)
        plt.pause(.001)

        # save stats separated from the stepping object
        step_stats.append(S.stats)

        # advance time
        S.time += S.dt

        # reset step (not necessary, but cleaner)
        S.reset_step()

        # initialize next step
        S.init_step(uend)


    # get exact values and print the error
    # uex = P.u_exact(S.params.Tend)
    # print(uex.pos.values,uend.pos.values)
    # print('Error:',np.linalg.norm(uex.pos.values-uend.pos.values,np.inf)/np.linalg.norm(uex.pos.values,np.inf))