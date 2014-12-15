from pySDC import Step as stepclass
from pySDC import CollocationClasses as collclass



import numpy as np
import copy as cp


from examples.heat1d.ProblemClass import heat1d
from examples.heat1d.TransferClass import mesh_to_mesh_1d
from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.Methods_Serial import mlsdc_step, sdc_step
import pySDC.Methods_Parallel as mp


if __name__ == "__main__":

    num_procs = 2

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
    description['collocation_class'] = collclass.CollGaussLegendre
    description['num_nodes'] = 3
    description['sweeper_class'] = imex_1st_order
    description['level_params'] = lparams
    description['transfer_class'] = mesh_to_mesh_1d


    MS = []
    for p in range(num_procs):
        MS.append(stepclass.step(sparams))
        MS[-1].generate_hierarchy(description)

    t0 = 0
    Tend = 0.25
    dt = 0.125

    P = MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    uend = mp.run_pfasst_serial(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

    exit()
    step_stats = []


    while MS[-1].time < Tend:

        uend = mlsdc_step(MS[0])

        step_stats.append(MS[0].stats)

        MS[0].time += MS[0].dt

        MS[0].reset_step()

        MS[0].init_step(uend)


    uex = P.u_exact(MS[0].time)

    print(step_stats[1].residual,step_stats[1].level_stats[0].residual)

    print('error at time %s: %s' %(MS[-1].time,np.linalg.norm(uex.values-uend.values,np.inf)/np.linalg.norm(
        uex.values,np.inf)))