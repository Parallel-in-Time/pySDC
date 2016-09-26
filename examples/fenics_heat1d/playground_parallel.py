import dolfin as df
import numpy as np
from mpi4py import MPI

import implementations.controller_classes.allinclusive_multigrid_MPI as mp
from examples.fenics_heat1d.ProblemClass import fenics_heat
from examples.fenics_heat1d.TransferClass import mesh_to_mesh_fenics
from implementations.datatype_classes import fenics_mesh,rhs_fenics_mesh
from implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC import CollocationClasses as collclass
from pySDC import Log
from pySDC.Stats import grep_stats, sort_stats
from pySDC.Step import step

if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    num_procs = 4

    # assert num_procs == 1,'turn on predictor!'

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 5E-09

    sparams = {}
    sparams['maxiter'] = 20

    # This comes as read-in for the problem class
    pparams = {}
    pparams['nu'] = 0.1
    pparams['t0'] = 0.0 # ugly, but necessary to set up ProblemClass
    # pparams['c_nvars'] = [(16,16)]
    pparams['c_nvars'] = [128]
    pparams['family'] = 'CG'
    pparams['order'] = [1]
    pparams['refinements'] = [1,0]


    # This comes as read-in for the transfer operations
    tparams = {}
    tparams['finter'] = True

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = fenics_heat
    description['problem_params'] = pparams
    description['dtype_u'] = fenics_mesh
    description['dtype_f'] = rhs_fenics_mesh
    description['collocation_class'] = collclass.CollGaussLegendre
    description['num_nodes'] = 3
    description['sweeper_class'] = imex_1st_order
    description['level_params'] = lparams
    description['transfer_class'] = mesh_to_mesh_fenics
    description['transfer_params'] = tparams

    # setup parameters "in time"
    t0 = 0
    dt = 0.5
    Tend = num_procs * dt

    # quickly generate block of steps
    S = step(sparams)
    S.generate_hierarchy(description)

    P = S.levels[0].prob
    uinit = P.u_exact(t0)

    uend, stats = mp.run_pfasst(S, uinit, t0, dt, Tend, comm)

    # df.plot(uend.values,interactive=True)

    # compute exact solution and compare
    uex = P.u_exact(Tend)

    print('(classical) error at time %s: %s' %(Tend,abs(uex-uend)/abs(uex)))


    uex = df.Expression('sin(a*x[0]) * cos(t)',a=np.pi,t=Tend)
    print('(fenics-style) error at time %s: %s' %(Tend,df.errornorm(uex,uend.values)))

    extract_stats = grep_stats(stats,iter=-1,type='residual')
    sortedlist_stats = sort_stats(extract_stats,sortby='step')
    print(extract_stats,sortedlist_stats)