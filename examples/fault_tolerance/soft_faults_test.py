
from pySDC import CollocationClasses as collclass

import numpy as np
import pickle as pkl

from examples.heat1d.ProblemClass import heat1d
from examples.heat1d.TransferClass import mesh_to_mesh_1d
from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order_softfaults import imex_1st_order
# import pySDC.PFASST_blockwise as mp
import pySDC.PFASST_stepwise as mp
# import pySDC.Methods as mp
from pySDC import Log
from pySDC.Stats import grep_stats, sort_stats

import pySDC.Plugins.fault_tolerance as ft
from pySDC.Plugins.visualization_tools import show_residual_across_simulation



if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    # logger = Log.setup_custom_logger('root')

    num_procs = 1

    # This comes as read-in for the level class  (this is optional!)
    lparams = {}
    lparams['restol'] = 1E-07

    # This comes as read-in for the step class (this is optional!)
    sparams = {}
    sparams['maxiter'] = 50
    sparams['fine_comm'] = True

    # This comes as read-in for the problem class
    pparams = {}
    pparams['nu'] = 0.5
    pparams['nvars'] = [255]

    # This comes as read-in for the transfer operations (this is optional!)
    tparams = {}
    tparams['finter'] = True

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = heat1d
    description['problem_params'] = pparams
    description['dtype_u'] = mesh
    description['dtype_f'] = rhs_imex_mesh
    description['collocation_class'] = collclass.CollGaussLobatto
    description['num_nodes'] = 3
    description['sweeper_class'] = imex_1st_order
    description['level_params'] = lparams
    # description['transfer_class'] = mesh_to_mesh_1d
    # description['transfer_params'] = tparams

    # setup parameters "in time"
    t0 = 0
    dt = 0.5
    Tend = num_procs*dt

    ft.soft_random = 0.0 # no faults
    ft.soft_do_correction = True

    # quickly generate block of steps
    MS = mp.generate_steps(num_procs,sparams,description)

    # get initial values on finest level
    P = MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend,stats = mp.run_pfasst(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

    # compute exact solution and compare
    uex = P.u_exact(Tend)
    err = np.linalg.norm(uex.values-uend.values,np.inf)/np.linalg.norm(uex.values,np.inf)
    # print('error at time %s: %s' %(Tend,err))

    extract_stats = grep_stats(stats,level=-1,type='residual')
    maxiter = 0
    for k,v in extract_stats.items():
        maxiter = max(maxiter,getattr(k,'iter'))
        # minres = min(minres,np.log10(v))
        # maxres = max(maxres,np.log10(v))
    # print(extract_stats)
    residuals = np.zeros(maxiter)
    for k,v in extract_stats.items():
        step = getattr(k,'step')
        iter = getattr(k,'iter')
        if iter is not -1 and step == num_procs-1:
            residuals[iter-1] = v

    ft.soft_stats.append((ft.soft_fault_injected,ft.soft_fault_detected,ft.soft_fault_hit,ft.soft_fault_missed,maxiter,residuals,err))
    ft.soft_fault_injected = 0
    ft.soft_fault_detected = 0
    ft.soft_fault_hit = 0
    ft.soft_fault_missed = 0
    # exit()

    ft.soft_random = 0.01 # 1% soft-faults
    ft.soft_safety_factor = 10.0
    ft.soft_do_correction = False

    nsim = 1000

    for k in range(nsim):
        print('-----------------------> Working on step %i of %i...' %(k+1,nsim))

        # quickly generate block of steps
        MS = mp.generate_steps(num_procs,sparams,description)

        # get initial values on finest level
        P = MS[0].levels[0].prob
        uinit = P.u_exact(t0)

        # call main function to get things done...
        uend,stats = mp.run_pfasst(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

        # compute exact solution and compare
        uex = P.u_exact(Tend)
        err = np.linalg.norm(uex.values-uend.values,np.inf)/np.linalg.norm(uex.values,np.inf)
        # print('error at time %s: %s' %(Tend,err))

        extract_stats = grep_stats(stats,level=-1,type='residual')
        maxiter = 0
        for k,v in extract_stats.items():
            maxiter = max(maxiter,getattr(k,'iter'))
            # minres = min(minres,np.log10(v))
            # maxres = max(maxres,np.log10(v))

        residuals = np.zeros(maxiter)
        for k,v in extract_stats.items():
            step = getattr(k,'step')
            iter = getattr(k,'iter')
            if iter is not -1 and step == num_procs-1:
                residuals[iter-1] = v

        ft.soft_stats.append((ft.soft_fault_injected,ft.soft_fault_detected,ft.soft_fault_hit,ft.soft_fault_missed,maxiter,residuals,err))
        ft.soft_fault_injected = 0
        ft.soft_fault_detected = 0
        ft.soft_fault_hit = 0
        ft.soft_fault_missed = 0

    # fname = 'HEAT_PFASST_soft_faults_nocorr_N1000_NOCOARSE.pkl'
    # fname = 'HEAT_PFASST_soft_faults_corr10x_N1000_NOCOARSE.pkl'
    # fname = 'HEAT_MLSDC_soft_faults_corr10x_N1000.pkl'
    # fname = 'HEAT_MLSDC_soft_faults_nocorr_N1000.pkl'

    fname = 'HEAT_SDC_soft_faults_nocorr_N1000.pkl'
    # fname = 'HEAT_SDC_soft_faults_corr1x_N1000.pkl'
    # fname = 'HEAT_SDC_soft_faults_corr5x_N1000.pkl'
    # fname = 'HEAT_SDC_soft_faults_corr10x_N1000.pkl'

    pkl.dump(ft.soft_stats, open(fname,'wb'))


