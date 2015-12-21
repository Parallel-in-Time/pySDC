from __future__ import print_function

from pySDC import CollocationClasses as collclass

import numpy as np
import matplotlib.pyplot as plt

from examples.fwsw.ProblemClass import swfw_scalar
from pySDC.datatype_classes.complex_mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order
import pySDC.PFASST_stepwise as mp
from pySDC import Log



if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 1
    
    N_s = 100
    N_f = 125

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-12

    sparams = {}
    sparams['maxiter'] = 4

    # This comes as read-in for the problem class
    pparams = {}
    pparams['lambda_s'] = 1j*np.linspace(0.0, 4.0, N_s)
    pparams['lambda_f'] = 1j*np.linspace(0.0, 8.0, N_f)
    pparams['u0'] = 1

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = swfw_scalar
    description['problem_params'] = pparams
    description['dtype_u'] = mesh
    description['dtype_f'] = rhs_imex_mesh
    description['collocation_class'] = collclass.CollGaussLobatto
    description['num_nodes'] = [3]
    description['do_LU'] = False
    description['sweeper_class'] = imex_1st_order
    description['level_params'] = lparams

    # quickly generate block of steps
    MS = mp.generate_steps(num_procs,sparams,description)

    # setup parameters "in time"
    t0   = 0
    dt   = 1.0
    Tend = 1.0

    P = MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    print('Init:',uinit.values)

    # call main function to get things done...
    uend,stats = mp.run_pfasst(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

    uex = P.u_exact(Tend)

    fig = plt.figure(figsize=(4,4))
    #pcol = plt.pcolor(pparams['lambda_s'].imag, pparams['lambda_f'].imag, np.absolute(uend.values).T, vmin=0.99, vmax=1.01)
    pcol = plt.pcolor(pparams['lambda_s'].imag, pparams['lambda_f'].imag, np.absolute(uend.values).T, vmin=1.0, vmax=2.0)
    # plt.pcolor(np.imag(uend.values))
    pcol.set_edgecolor('face')
    plt.plot([0, 4], [0, 4], color='w', linewidth=2.5)
    #plt.colorbar()
    plt.gca().set_xticks([0.0, 1.0, 2.0, 3.0])
    plt.xlabel('$\Delta t \lambda_{slow}$', fontsize=18, labelpad=0.0)
    plt.ylabel('$\Delta t \lambda_{fast}$', fontsize=18)

    #plt.show()
    fig.savefig('sdc-fwsw-stability-K'+str(sparams['maxiter'])+'-M'+str(description['num_nodes'][0])+'.pdf', bbox_inches='tight')
    print('error at time %s: %s' %(Tend,np.linalg.norm(uex.values-uend.values,np.inf)/np.linalg.norm(uex.values,
                                                                                                     np.inf)))

