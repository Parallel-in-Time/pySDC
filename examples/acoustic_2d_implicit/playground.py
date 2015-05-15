
from pySDC import CollocationClasses as collclass

import numpy as np

from ProblemClass import acoustic_2d_implicit
#from examples.sharpclaw_burgers1d.TransferClass import mesh_to_mesh_1d
from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order
import pySDC.PFASST_stepwise as mp
from pySDC import Log
from pySDC.Stats import grep_stats, sort_stats

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from unflatten import unflatten

if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 1

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 3E-12

    sparams = {}
    sparams['maxiter'] = 50

    # setup parameters "in time"
    t0 = 0
    dt = 0.1
    Tend = 5*dt

    # This comes as read-in for the problem class
    pparams = {}
    pparams['nvars'] = [(3,150,25)]

    # This comes as read-in for the transfer operations
    #tparams = {}
    #tparams['finter'] = False

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class']     = acoustic_2d_implicit
    description['problem_params']    = pparams
    description['dtype_u']           = mesh
    description['dtype_f']           = rhs_imex_mesh
    description['collocation_class'] = collclass.CollGaussLobatto
    description['num_nodes']         = 3
    description['sweeper_class']     = imex_1st_order
    description['level_params']      = lparams
    #description['transfer_class'] = mesh_to_mesh_1d
    #description['transfer_params'] = tparams

    # quickly generate block of steps
    MS = mp.generate_steps(num_procs,sparams,description)

    # get initial values on finest level
    P = MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend,stats = mp.run_pfasst(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

    # compute exact solution and compare
    uex = P.u_exact(Tend)

    print('error at time %s: %s' %(Tend,np.linalg.norm(uex.values.flatten()-uend.values.flatten(),np.inf)/np.linalg.norm(
        uex.values.flatten(),np.inf)))

    fig = plt.figure(figsize=(8,8))
    fig.clear()
    yplot = uend.values
    ax = fig.gca(projection='3d')
    ax.view_init(elev=0., azim=-90.)
    surf = ax.plot_surface(P.xx, P.zz, yplot[2,:,:], rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlim(left   =  P.x_b[0], right = P.x_b[1])
    ax.set_ylim(bottom =  P.z_b[0], top   = P.z_b[1])
    ax.set_zlim(bottom = -1.0, top   = 1.0)
    plt.xlabel('x')
    plt.ylabel('z')
    plt.show()

    # extract_stats = grep_stats(stats,iter=-1,type='residual')
    # sortedlist_stats = sort_stats(extract_stats,sortby='step')
    # print(extract_stats,sortedlist_stats)