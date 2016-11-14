import numpy as np
import pySDC.core.PFASST_stepwise as mp
import pySDC.helpers.fault_tolerance as ft
from pySDC.core.Stats import grep_stats
from pySDC.core.datatype_classes.fenics_mesh import fenics_mesh
from pySDC.core.sweeper_classes.generic_LU import generic_LU

from examples.fenics_grayscott.TransferClass import mesh_to_mesh_fenics
from pySDC.core import CollocationClasses as collclass
from pySDC.core import Log
from pySDC.implementations.problem_classes.ProblemClass import fenics_grayscott


if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 32

    # assert num_procs == 1,'turn on predictor!'

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-07

    sparams = {}
    sparams['maxiter'] = 50
    sparams['fine_comm'] = True

    # This comes as read-in for the problem class
    pparams = {}
    # pparams['Du'] = 1.0
    # pparams['Dv'] = 0.01
    # pparams['A'] = 0.01
    # pparams['B'] = 0.10
    # splitting pulses until steady state
    # pparams['Du'] = 1.0
    # pparams['Dv'] = 0.01
    # pparams['A'] = 0.02
    # pparams['B'] = 0.079
    # splitting pulses until steady state
    pparams['Du'] = 1.0
    pparams['Dv'] = 0.01
    pparams['A'] = 0.09
    pparams['B'] = 0.086

    pparams['t0'] = 0.0 # ugly, but necessary to set up ProblemClass
    # pparams['c_nvars'] = [(16,16)]
    pparams['c_nvars'] = [512]
    pparams['family'] = 'CG'
    pparams['order'] = [4]
    pparams['refinements'] = [1,0]


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
    # description['hook_class'] = fenics_output

    ft.hard_random = 0.03

    # strategies = ['NOFAULT']
    # strategies = ['SPREAD','INTERP','INTERP_PREDICT','SPREAD_PREDICT']
    strategies = ['INTERP', 'INTERP_PREDICT']

    for strategy in strategies:

        print('------------------------------------------ working on strategy ',strategy)
        ft.strategy = strategy

        # read in reference data from clean run, will provide reproducable locations for faults
        if not strategy is 'NOFAULT':
            reffile = np.load('PFASST_GRAYSCOTT_stats_hf_NOFAULT_P32.npz')
            ft.refdata = reffile['hard_stats']

        # quickly generate block of steps
        MS = mp.generate_steps(num_procs,sparams,description)

        # setup parameters "in time"
        t0 = MS[0].levels[0].prob.t0
        dt = 2.0
        Tend = 1280.0

        # get initial values on finest level
        P = MS[0].levels[0].prob
        uinit = P.u_exact(t0)

        # call main function to get things done...
        uend,stats = mp.run_pfasst(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

        # stats magic: get all residuals
        extract_stats = grep_stats(stats,level=-1,type='residual')

        # find boundaries
        maxsteps = 0
        maxiter = 0
        minres = 0
        maxres = -99
        for k,v in extract_stats.items():
            maxsteps = max(maxsteps,getattr(k,'step'))
            maxiter = max(maxiter,getattr(k,'iter'))
            minres = min(minres,np.log10(v))
            maxres = max(maxres,np.log10(v))

        # fill residual array
        residual = np.zeros((maxiter,maxsteps+1))
        residual[:] = 0
        for k,v in extract_stats.items():
            step = getattr(k,'step')
            iter = getattr(k,'iter')
            if iter is not -1:
                residual[iter-1,step] = v

        # stats magic: get niter (probably redundant with maxiter)
        extract_stats = grep_stats(stats,iter=-1,type='niter')
        iter_count = np.zeros(maxsteps+1)
        for k,v in extract_stats.items():
            step = getattr(k,'step')
            iter_count[step] = v
        print(iter_count)

        np.savez('PFASST_GRAYSCOTT_stats_hf_'+ft.strategy+'_P'+str(num_procs)+'_cN512',residual=residual,iter_count=iter_count,hard_stats=ft.hard_stats)

