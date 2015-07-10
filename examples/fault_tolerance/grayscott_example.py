
from pySDC import CollocationClasses as collclass

from examples.fenics_grayscott.ProblemClass import fenics_grayscott
from pySDC.datatype_classes.fenics_mesh import fenics_mesh
from examples.fenics_grayscott.TransferClass import mesh_to_mesh_fenics
from examples.fenics_grayscott.HookClass import fenics_output
from pySDC.sweeper_classes.generic_LU import generic_LU
# import pySDC.PFASST_blockwise as mp
import pySDC.PFASST_stepwise as mp
from pySDC import Log
from pySDC.Stats import grep_stats, sort_stats

import dolfin as df
import numpy as np

import pySDC.Plugins.fault_tolerance as ft


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
    pparams['c_nvars'] = [256]
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

    # strategies = ['INTERP']
    strategies = ['SPREAD','INTERP','INTERP_PREDICT','SPREAD_PREDICT','NOFAULT']

    for strategy in strategies:

        print('------------------------------------------ working on strategy ',strategy)
        ft.strategy = strategy

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

        extract_stats = grep_stats(stats,type='residual')

        maxsteps = 0
        maxiter = 0
        minres = 0
        maxres = -99
        for k,v in extract_stats.items():
            maxsteps = max(maxsteps,getattr(k,'step'))
            maxiter = max(maxiter,getattr(k,'iter'))
            minres = min(minres,np.log10(v))
            maxres = max(maxres,np.log10(v))
            # print(getattr(k,'step'),getattr(k,'iter'),v)

        # print(maxsteps,maxiter,minres,maxres)

        residual = np.zeros((maxiter,maxsteps+1))
        residual[:] = 0

        for k,v in extract_stats.items():
            step = getattr(k,'step')
            iter = getattr(k,'iter')
            if iter is not -1:
                residual[iter-1,step] = v

        extract_stats = grep_stats(stats,iter=-1,type='niter')
        iter_count = np.zeros(maxsteps+1)
        for k,v in extract_stats.items():
            step = getattr(k,'step')
            iter_count[step] = v
        print(iter_count)

        np.savez('GRAYSCOTT_stats_hf_'+ft.strategy+'_new',residual=residual,iter_count=iter_count,hard_stats=ft.hard_stats)

    # u1,u2 = df.split(uend.values)
    # df.plot(u1,interactive=True)

