import numpy as np
from mpi4py import MPI

from pySDC.controller_classes.PFASST_blockwise_parallel import PFASST_blockwise_parallel
from examples.heat1d.ProblemClass import heat1d
from examples.heat1d.TransferClass import mesh_to_mesh_1d
from pySDC import CollocationClasses as collclass
from pySDC import Log
from pySDC.Step import step
from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order



if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    comm = MPI.COMM_WORLD

    # This comes as read-in for the level class  (this is optional!)
    lparams = {}
    lparams['restol'] = 1E-10

    # This comes as read-in for the step class (this is optional!)
    sparams = {}
    sparams['maxiter'] = 20
    sparams['fine_comm'] = True
    sparams['predict'] = True

    # This comes as read-in for the problem class
    pparams = {}
    pparams['nu'] = 1.0
    pparams['nvars'] = [63,31]

    # This comes as read-in for the transfer operations (this is optional!)
    tparams = {}
    tparams['finter'] = False
    tparams['iorder'] = 6
    tparams['rorder'] = 2

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = heat1d
    description['problem_params'] = pparams
    description['dtype_u'] = mesh
    description['dtype_f'] = rhs_imex_mesh
    description['collocation_class'] = collclass.CollGaussRadau_Right
    description['num_nodes'] = 5
    description['sweeper_class'] = imex_1st_order
    description['level_params'] = lparams
    description['transfer_class'] = mesh_to_mesh_1d
    description['transfer_params'] = tparams

    # quickly generate block of steps
    # MS = mp.generate_steps(num_procs,sparams,description)

    # setup parameters "in time"
    t0 = 0
    dt = 0.1
    Tend = 3*dt

    S = step(sparams)
    S.generate_hierarchy(description)

    P = S.levels[0].prob
    uinit = P.u_exact(t0)

    PFASST = PFASST_blockwise_parallel(S=S,comm=comm)
    uend, stats = PFASST.run(u0=uinit,t0=t0,dt=dt,Tend=Tend)

    # compute exact solution and compare
    num_procs = comm.Get_size()
    rank = comm.Get_rank()
    if rank == 0:
        uex = P.u_exact(Tend)

        print('error at time %s: %s' % (Tend, np.linalg.norm(uex.values - uend.values, np.inf) / np.linalg.norm(
            uex.values, np.inf)))


