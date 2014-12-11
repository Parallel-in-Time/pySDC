from pySDC import Step as stepclass
from pySDC import Level as levclass
from pySDC import CollocationClasses as collclass

import numpy as np

from examples.heat1d.ProblemClass import heat1d
from examples.heat1d.TransferClass import mesh_to_mesh_1d
from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.Methods import sdc_step, mlsdc_step


if __name__ == "__main__":

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-12

    sparams = {}
    sparams['Tend'] = 2*0.125
    sparams['maxiter'] = 10

    # This comes as read-in for the problem class
    cparams_f = {}
    cparams_f['nu'] = 0.1
    cparams_f['nvars'] = 255

    cparams_m = {}
    cparams_m['nu'] = 0.1
    cparams_m['nvars'] = 127
    cparams_m['dx'] = 1/(cparams_m['nvars']+1)

    cparams_c = {}
    cparams_c['nu'] = 0.1
    cparams_c['nvars'] = 63
    cparams_c['dx'] = 1/(cparams_c['nvars']+1)

    L0 = levclass.level(problem_class       =   heat1d,
                        problem_params      =   cparams_f,
                        dtype_u             =   mesh,
                        dtype_f             =   rhs_imex_mesh,
                        collocation_class   =   collclass.CollGaussLegendre,
                        num_nodes           =   3,
                        sweeper_class       =   imex_1st_order,
                        level_params        =   lparams,
                        id                  =   'L0')

    L1 = levclass.level(problem_class       =   heat1d,
                        problem_params      =   cparams_m,
                        dtype_u             =   mesh,
                        dtype_f             =   rhs_imex_mesh,
                        collocation_class   =   collclass.CollGaussLegendre,
                        num_nodes           =   3,
                        sweeper_class       =   imex_1st_order,
                        level_params        =   lparams,
                        id                  =   'L1')

    # L2 = levclass.level(problem_class       =   heat1d,
    #                     problem_params      =   cparams_c,
    #                     dtype_u             =   mesh,
    #                     dtype_f             =   rhs_imex_mesh,
    #                     collocation_class   =   collclass.CollGaussLegendre,
    #                     num_nodes           =   3,
    #                     sweeper_class       =   imex_1st_order,
    #                     level_params        =   lparams,
    #                     id                  =   'L2')

    S = stepclass.step(sparams)
    S.register_level(L0)
    S.register_level(L1)
    # S.register_level(L2)


    S.connect_levels(transfer_class = mesh_to_mesh_1d,
                     fine_level     = L0,
                     coarse_level   = L1)

    # S.connect_levels(transfer_class = mesh_to_mesh_1d,
    #                  fine_level     = L1,
    #                  coarse_level   = L2)

    # del L0,L1,L2
    # del lparams,sparams,cparams_f,cparams_m,cparams_c

    S.time = 0
    S.dt = 0.125

    S.stats.niter = 0

    P = S.levels[0].prob
    uinit = P.u_exact(S.time)

    S.init_step(uinit)

    step_stats = []

    nsteps = int(S.params.Tend/S.dt)

    for n in range(nsteps):

        uend = mlsdc_step(S)

        step_stats.append(S.stats)

        S.time += S.dt

        S.reset_step()

        S.init_step(uend)


    uex = P.u_exact(S.time)

    print(step_stats[1].residual,step_stats[1].level_stats[0].residual)

    print('error at time %s: %s' %(S.time,np.linalg.norm(uex.values-uend.values,np.inf)/np.linalg.norm(uex.values,np.inf)))

