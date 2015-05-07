
from pySDC import CollocationClasses as collclass

import numpy as np

from ProblemClass import acoustic_2d_implicit
from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order
import pySDC.Methods as mp
from pySDC import Log
from pySDC.Stats import grep_stats, sort_stats

# Sharpclaw imports
from clawpack import pyclaw
from clawpack import riemann
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 1

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1E-10

    sparams = {}
    sparams['maxiter'] = 5

    # setup parameters "in time"
    t0 = 0
    dt = 0.05
    Tend = 20*dt

    # This comes as read-in for the problem class
    pparams = {}
    pparams['nvars'] = [(3, 250, 4)]
    pparams['cs']    = 1.0
    
    # This comes as read-in for the transfer operations
    tparams = {}
    tparams['finter'] = True

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class']     = acoustic_2d_implicit
    description['problem_params']    = pparams
    description['dtype_u']           = mesh
    description['dtype_f']           = rhs_imex_mesh
    description['collocation_class'] = collclass.CollGaussLobatto
    description['num_nodes']         = 5
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
    uend,stats = mp.run_pfasst_serial(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

    # compute exact solution and compare
    uex = P.u_exact(Tend)

    diff = uex.values[2,:,:] - uend.values[2,:,:]
    error = np.linalg.norm( diff.flatten(), np.inf )
    print('error at time %s: %s' % (Tend,error))

#print np.shape(P.domainx)
#print np.shape(P.domainz)
#print np.shape(uex.values)
    fig = plt.figure(figsize=(18,6))
    #    ax = fig.gca(projection='3d')
    #ax.view_init(elev=90., azim=90.)
#plt.contourf(P.domainx, P.domainz, uex.values[2,:,:])
#    surf = ax.plot_surface(P.domainx, P.domainz, uend.values[2,:,:], rstride=4, cstride=4, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    plt.plot(np.linspace(0,1,250), uex.values[2,:,2], 'r')
    plt.plot(np.linspace(0,1,250), uend.values[2,:,2], 'b')
    plt.xlabel('x')
    plt.ylabel('z')
#plt.axes().set_aspect('equal')    #ax.set_xlim3d(x[0], x[Nx-1])
    plt.show()