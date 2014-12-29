
from pySDC import CollocationClasses as collclass

import numpy as np

from examples.heat1d.ProblemClass import heat1d
from examples.heat1d.TransferClass import mesh_to_mesh_1d
from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order
import pySDC.Methods_Parallel as mp


if __name__ == "__main__":

    num_procs = 3

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-12

    sparams = {}
    sparams['maxiter'] = 10

    # This comes as read-in for the problem class
    pparams = {}
    pparams['nu'] = 0.1
    pparams['nvars'] = [255,127]

    description = {}
    description['problem_class'] = heat1d
    description['problem_params'] = pparams
    description['dtype_u'] = mesh
    description['dtype_f'] = rhs_imex_mesh
    description['collocation_class'] = collclass.CollGaussLobatto
    description['num_nodes'] = 3
    description['sweeper_class'] = imex_1st_order
    description['level_params'] = lparams
    description['transfer_class'] = mesh_to_mesh_1d

    MS = mp.generate_steps(num_procs,sparams,description)

    t0 = 0
    Tend = 0.5
    dt = 0.125

    P = MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    uend,step_stats = mp.run_pfasst_serial(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

    uex = P.u_exact(Tend)
    print('error at time %s: %s' %(Tend,np.linalg.norm(uex.values-uend.values,np.inf)/np.linalg.norm(
        uex.values,np.inf)))
    print(len(step_stats),len(step_stats[1].level_stats),len(step_stats[-1].level_stats[0].iter_stats))
    print(step_stats[1].residual,step_stats[0].level_stats[0].iter_stats[0].residual)
