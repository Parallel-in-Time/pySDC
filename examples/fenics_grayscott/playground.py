import pySDC.core.deprecated.PFASST_blockwise_old as mp

from examples.fenics_grayscott.HookClass import fenics_output
from examples.fenics_grayscott.TransferClass import mesh_to_mesh_fenics
from pySDC.core import CollocationClasses as collclass
from pySDC.core import Log
from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh
from pySDC.implementations.problem_classes.ProblemClass import fenics_grayscott
from pySDC.implementations.sweeper_classes.generic_LU import generic_LU

if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 1

    # assert num_procs == 1,'turn on predictor!'

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-07

    sparams = {}
    sparams['maxiter'] = 20
    sparams['fine_comm'] = True

    # This comes as read-in for the problem class
    pparams = {}
    pparams['Du'] = 1.0
    pparams['Dv'] = 0.01
    pparams['A'] = 0.01
    pparams['B'] = 0.10
    # pparams['Du'] = 2E-05
    # pparams['Dv'] = 1E-05
    # pparams['A'] = 0.03
    # pparams['B'] = 0.092

    pparams['t0'] = 0.0 # ugly, but necessary to set up ProblemClass
    # pparams['c_nvars'] = [(16,16)]
    pparams['c_nvars'] = [256]
    pparams['family'] = 'CG'
    pparams['order'] = [4]
    pparams['refinements'] = [1]


    # This comes as read-in for the transfer operations
    tparams = {}
    tparams['finter'] = True

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = fenics_grayscott
    description['problem_params'] = pparams
    description['dtype_u'] = fenics_mesh
    description['dtype_f'] = fenics_mesh
    description['collocation_class'] = collclass.CollGaussRadau_Right
    description['num_nodes'] = 3
    description['sweeper_class'] = generic_LU
    description['level_params'] = lparams
    description['transfer_class'] = mesh_to_mesh_fenics
    description['transfer_params'] = tparams
    description['hook_class'] = fenics_output

    # quickly generate block of steps
    MS = mp.generate_steps(num_procs,sparams,description)

    # setup parameters "in time"
    t0 = MS[0].levels[0].prob.t0
    dt = 5.0
    Tend = 400

    # get initial values on finest level
    P = MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend,stats = mp.run_pfasst(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

    # u1,u2 = df.split(uend.values)
    # df.plot(u1,interactive=True)

