import sys
sys.path.append("/private/PycharmProjects/pySDC")
import numpy as np
import scipy as sp
import scipy.linalg as la

import matplotlib.pyplot as plt
import matplotlib as mplt
import pylab

# %matplotlib inline
# Own collection of tools, require the imports from above
# from pint_matrix_tools import *

from pySDC import CollocationClasses as collclass
from examples.matrix_test_eq.ProblemClass import test_eq
from examples.matrix_test_eq.TransferClass import value_to_value

from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order
import pySDC.PFASST_blockwise as mp
from pySDC import Log
from pySDC.Stats import grep_stats, sort_stats
import pySDC.MatrixMethods as mmp

from examples.heat1d_periodic.HookClass import error_output

def generate_linpfasst(opts,uinit=False,debug=False):
    # make empty class
    opt = mmp.Bunch()
    for k, v in opts.items():
        setattr(opt, k, v)
    t0 = opt.t0
    dt = opt.dt
    Tend = opt.num_procs * dt

    # quickly generate block of steps
    MS = mp.generate_steps(opt.num_procs, opt.sparams, opt.description)
    P = MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    # initial ordering of the steps: 0,1,...,Np-1
    slots = [p for p in range(opt.num_procs)]

    # initialize time variables of each step
    for p in slots:
        MS[p].status.dt = dt # could have different dt per step here
        MS[p].status.time = t0 + sum(MS[j].status.dt for j in range(p))
        MS[p].status.step = p

    # initialize linear_pfasst
    transfer_list = mmp.generate_transfer_list(MS, opt.description['transfer_class'], **opt.tparams)
    lin_pfasst = mmp.generate_LinearPFASST(MS, transfer_list, uinit.values, **opt.tparams)

    return lin_pfasst,MS,transfer_list

if __name__ == "__main__":
    # number of processors and also number of subintervals
    num_procs = 4

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-10

    sparams = {}
    sparams['maxiter'] = 20
    # turns on and of the communication between fine and coarse level
    sparams['fine_comm'] = True
    #
    sparams['predict'] = False

    # This is doing the martin weiser trick for pfasst
    swparams = {}
    swparams['do_LU'] = True


    # This comes as read-in for the problem class
    pparams = {}
    pparams['lamb'] = -20
    # use two identical levels in space
    pparams['nvars'] = [1,1]

    # This comes as read-in for the all kind of generating options for the matrix classes
    mparams = {}
    mparams['sparse_format'] = "array"

    # This comes as read-in for the transfer operations and the preconditioner
        # This comes as read-in for the transfer operations
    tparams = {}
    tparams['finter'] = False
    tparams['sparse_format'] = "array"
    tparams['interpolation_order'] = [[1, 1]]*num_procs
    tparams['restriction_order'] = [[1, 1]]*num_procs
    tparams['t_interpolation_order'] = [2]*num_procs
    tparams['t_restriction_order'] = [2]*num_procs
    # This is doing the martin weiser trick for linearpfasst not needed anymore
    tparams['q_precond'] = 'QI'

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = test_eq
    description['problem_params'] = pparams
    description['dtype_u'] = mesh
    description['dtype_f'] = rhs_imex_mesh
#     description['collocation_class'] = collclass.CollGaussLobatto
    description['collocation_class'] = collclass.CollGaussRadau_Right_LU_Trick
    description['num_nodes'] = [5, 3]
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = swparams
    description['level_params'] = lparams
#     description['transfer_class'] = mesh_to_mesh_1d_periodic
    description['transfer_class'] = value_to_value
    description['transfer_params'] = tparams
    description['hook_class'] = error_output

    # Options for run_linear_pfasst
    linpparams = {}
    linpparams['run_type'] = "tolerance"
    linpparams['norm'] = lambda x: np.linalg.norm(x, np.inf)
    linpparams['tol'] = lparams['restol']

    # initial time
    t0 = 0.0
    dt = 0.1
    Tend = num_procs*dt

    opts = {'description': description, 'linpparams': linpparams, 'tparams': tparams,
            'mparams': mparams, 'pparams': pparams, 'sparams': sparams, 'lparams': lparams,
            'num_procs': num_procs, 't0': t0, 'dt': dt}

    opt = mmp.Bunch()
    for k, v in opts.items():
        setattr(opt, k, v)

    lin_pfasst, MS, transfer_list = generate_linpfasst(opts)
    # MS[0].levels[0].sweep.coll.nodes
