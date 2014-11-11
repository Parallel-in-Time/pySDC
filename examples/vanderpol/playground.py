from pySDC import Step as stepclass
from pySDC import Level as levclass
from pySDC import CollocationClasses as collclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
from subprocess import call

from examples.vanderpol.ProblemClass import vanderpol
from pySDC.sweeper_classes.generic_LU import generic_LU
from pySDC.datatype_classes.mesh import mesh
from pySDC.Methods import adaptive_sdc_step, sdc_step


if __name__ == "__main__":

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-10

    sparams = {}
    sparams['Tend'] = 15.0
    sparams['maxiter'] = 100
    sparams['pred_iter_lim'] = 8

    # This comes as read-in for the problem class
    cparams = {}
    cparams['newton_tol'] = 1E-12
    cparams['maxiter'] = 50
    cparams['mu'] = 5
    cparams['u0'] = np.array([2.0,0])

    L0 = levclass.level(problem_class       =   vanderpol,
                        problem_params      =   cparams,
                        dtype_u             =   mesh,
                        dtype_f             =   mesh,
                        collocation_class   =   collclass.CollGaussLegendre,
                        num_nodes           =   3,
                        sweeper_class       =   generic_LU,
                        level_params        =   lparams,
                        id                  =   'L0')


    S = stepclass.step(sparams)
    S.register_level(L0)

    S.time = 0
    S.dt = 0.1
    S.stats.niter = 0

    P = S.levels[0].prob
    uinit = P.u_exact(S.time)

    S.init_step(uinit)

    print('Init:',S.levels[0].u[0].values)

    fig = plt.figure(figsize=(10,10))
    # plt.ion()
    plt.axis([-2.5, 2.5, -10.5, 10.5])

    # hl, = plt.plot(S.levels[0].u[0].values[0],S.levels[0].u[0].values[1],'bo')


    nsteps = int(S.params.Tend/S.dt)

    step_stats = []

    nsteps = int(S.params.Tend/S.dt)
    xdata = []
    ydata = []
    cdata = []
    while S.time < S.params.Tend:

        uend = adaptive_sdc_step(S)

        step_stats.append(S.stats)

        xdata.append(uend.values[0])
        ydata.append(uend.values[1])
        cdata.append(cm.jet(S.dt))
        # hl.set_xdata(np.append(hl.get_xdata(),uend.values[0]))
        # hl.set_ydata(np.append(hl.get_ydata(),uend.values[1]))
        # hl.set_color(np.append(hl.get_color(),cm.hot(S.dt)))
        # plt.draw()

        S.time += S.dt

        S.reset_step()

        S.init_step(uend)

    print('u_end:',uend.values)

    print('Min/Max/Sum number of iterations: %s/%s/%s' %(min(stats.niter for stats in step_stats),
                                                         max(stats.niter for stats in step_stats),
                                                         sum(stats.niter for stats in step_stats)))

    plt.scatter(xdata,ydata,c=cdata,s=40,label='Stepsize')
    plt.legend()
    plt.show()

    # plt.grid('on')
    # plt.tight_layout()
    #
    # name = 'vanderpol_traj.pdf'
    # plt.savefig(name,rasterized=True)
    # call('pdfcrop '+name+' '+name,shell=True)
    #
    # plt.show()