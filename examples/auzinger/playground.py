from pySDC import Step as stepclass
from pySDC import Level as levclass
from pySDC import CollocationClasses as collclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from subprocess import call

from examples.auzinger.ProblemClass import auzinger
from pySDC.datatype_classes.mesh import mesh
from pySDC.sweeper_classes.generic_LU import generic_LU
from pySDC.Methods import sdc_step, mlsdc_step


if __name__ == "__main__":

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-10

    sparams = {}
    sparams['Tend'] = 0.01
    sparams['maxiter'] = 20

    # This comes as read-in for the problem class
    cparams = {}
    cparams['newton_tol'] = 1E-12
    cparams['maxiter'] = 50

    L0 = levclass.level(problem_class       =   auzinger,
                        problem_params      =   cparams,
                        dtype_u             =   mesh,
                        dtype_f             =   mesh,
                        collocation_class   =   collclass.CollGaussLobatto,
                        num_nodes           =   5,
                        sweeper_class       =   generic_LU,
                        level_params        =   lparams,
                        id                  =   'L0')


    S = stepclass.step(sparams)
    S.register_level(L0)

    S.time = 0
    S.dt = 0.01
    S.stats.niter = 0

    P = S.levels[0].prob
    uinit = P.u_exact(S.time)

    S.init_step(uinit)

    print('Init:',S.levels[0].u[0].values)

    # fig = plt.figure(figsize=(10,10))
    # plt.ion()
    # plt.axis([-2.5, 2.5, -10.5, 10.5])
    #
    # hl, = plt.plot(S.levels[0].u[0].values[0],S.levels[0].u[0].values[1],'b-')


    nsteps = int(S.params.Tend/S.dt)

    step_stats = []

    nsteps = int(S.params.Tend/S.dt)

    for n in range(nsteps):

        uend = sdc_step(S)

        step_stats.append(S.stats)

        # hl.set_xdata(np.append(hl.get_xdata(),uend.values[0]))
        # hl.set_ydata(np.append(hl.get_ydata(),uend.values[1]))
        # plt.draw()

        S.time += S.dt

        S.reset_step()

        S.init_step(uend)

    uex = P.u_exact(S.time)
    print(uend.values)
    print(uex.values)
    print('Error:',np.linalg.norm(uex.values-uend.values,np.inf)/np.linalg.norm(uex.values,np.inf))


    # plt.grid('on')
    # plt.tight_layout()
    #
    # name = 'vanderpol_traj.pdf'
    # plt.savefig(name,rasterized=True)
    # call('pdfcrop '+name+' '+name,shell=True)
    #
    # plt.show()