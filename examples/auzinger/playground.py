from pySDC import Step as stepclass
from pySDC import Level as levclass
from pySDC import CollocationClasses as collclass

import numpy as np

from examples.auzinger.ProblemClass import auzinger
from pySDC.datatype_classes.mesh import mesh
from pySDC.sweeper_classes.generic_LU import generic_LU
from pySDC.Methods import sdc_step, mlsdc_step, adaptive_sdc_step


if __name__ == "__main__":

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-10

    sparams = {}
    sparams['Tend'] = 20.0
    sparams['maxiter'] = 20
    sparams['pred_iter_lim'] = 4


    # This comes as read-in for the problem class
    cparams = {}
    cparams['newton_tol'] = 1E-12
    cparams['maxiter'] = 50

    L0 = levclass.level(problem_class       =   auzinger,
                        problem_params      =   cparams,
                        dtype_u             =   mesh,
                        dtype_f             =   mesh,
                        collocation_class   =   collclass.CollGaussLobatto,
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


    nsteps = int(S.params.Tend/S.dt)

    step_stats = []

    while S.time < S.params.Tend:

        uend = adaptive_sdc_step(S)

        step_stats.append(S.stats)

        S.time += S.dt

        S.reset_step()

        S.init_step(uend)

    uex = P.u_exact(S.time)

    print('Error:',np.linalg.norm(uex.values-uend.values,np.inf)/np.linalg.norm(uex.values,np.inf))

    print('Min/Max/Sum number of iterations: %s/%s/%s' %(min(stats.niter for stats in step_stats),
                                                         max(stats.niter for stats in step_stats),
                                                         sum(stats.niter for stats in step_stats)))
