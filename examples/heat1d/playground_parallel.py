import numpy as np
from mpi4py import MPI

from pySDC.implementations.problem_classes.HeatEquation_1D_FD_forced import heat1d_forced
from pySDC.implementations.transfer_classes.TransferMesh_1D_IMEX import mesh_to_mesh_1d_dirichlet
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.allinclusive_multigrid_MPI import allinclusive_multigrid_MPI
from pySDC.implementations.controller_classes.allinclusive_classic_MPI import allinclusive_classic_MPI
from pySDC.implementations.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC import Log

if __name__ == "__main__":

    comm = MPI.COMM_WORLD

    # This comes as read-in for the level class  (this is optional!)
    lparams = {}
    lparams['restol'] = 1E-10
    lparams['dt'] = 0.12

    # This comes as read-in for the controller
    cparams = {}
    cparams['fine_comm'] = True
    cparams['predict'] = True
    cparams['logger_level'] = 20

    # This comes as read-in for the problem class
    pparams = {}
    pparams['nu'] = 1.0
    pparams['freq'] = 1.0
    pparams['nvars'] = [1023,511]

    # This comes as read-in for the transfer operations (this is optional!)
    tparams = {}
    # tparams['finter'] = False
    tparams['iorder'] = 6
    tparams['rorder'] = 2

    # This comes as read-in for the sweeper class
    swparams = {}
    swparams['collocation_class'] = CollGaussRadau_Right
    swparams['num_nodes'] = 5

    # step parameters
    sparams = {}
    sparams['maxiter'] = 20

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = heat1d_forced
    description['problem_params'] = pparams
    description['dtype_u'] = mesh
    description['dtype_f'] = rhs_imex_mesh
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = swparams
    description['level_params'] = lparams
    description['step_params'] = sparams
    description['space_transfer_class'] = mesh_to_mesh_1d_dirichlet
    description['space_transfer_params'] = tparams

    # initialize controller
    PFASST = allinclusive_multigrid_MPI(controller_params=cparams, description=description, comm=comm)
    # PFASST = allinclusive_classic_MPI(controller_params=cparams, description=description, comm=comm)

    # setup parameters "in time"
    t0 = 0
    Tend = 3*0.12

    # set initial condition
    P = PFASST.S.levels[0].prob
    uinit = P.u_exact(t0)

    uend, stats = PFASST.run(u0=uinit,t0=t0,Tend=Tend)

    # compute exact solution and compare
    num_procs = comm.Get_size()
    rank = comm.Get_rank()
    if rank == 0:
        uex = P.u_exact(Tend)

        print('error at time %s: %s' % (Tend, np.linalg.norm(uex.values - uend.values, np.inf) / np.linalg.norm(
            uex.values, np.inf)))


