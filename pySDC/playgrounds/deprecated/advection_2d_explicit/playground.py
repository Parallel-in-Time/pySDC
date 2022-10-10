import pySDC.core.Methods as mp
from ProblemClass import advection_2d_explicit
from matplotlib import pyplot as plt

from pySDC.core import CollocationClasses as collclass
from pySDC.core import Log
from pySDC.implementations.datatype_classes import mesh, rhs_imex_mesh
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.playgrounds.deprecated.advection_2d_explicit import plot_solution

if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 1

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 1e-10

    sparams = {}
    sparams['maxiter'] = 0

    # setup parameters "in time"
    t0 = 0
    dt = 0.001
    Tend = 100 * dt

    # This comes as read-in for the problem class
    pparams = {}
    pparams['nvars'] = [(1, 100, 50)]

    # This comes as read-in for the transfer operations
    tparams = {}
    tparams['finter'] = True

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = advection_2d_explicit
    description['problem_params'] = pparams
    description['dtype_u'] = mesh
    description['dtype_f'] = rhs_imex_mesh
    description['collocation_class'] = collclass.CollGaussLobatto
    description['num_nodes'] = 2
    description['sweeper_class'] = imex_1st_order
    description['level_params'] = lparams
    description['hook_class'] = plot_solution
    # description['transfer_class'] = mesh_to_mesh
    # description['transfer_params'] = tparams

    # quickly generate block of steps
    MS = mp.generate_steps(num_procs, sparams, description)

    # get initial values on finest level
    P = MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = mp.run_pfasst_serial(MS, u0=uinit, t0=t0, dt=dt, Tend=Tend)

    # compute exact solution and compare
    uex = P.u_exact(Tend)

    # print('error at time %s: %s' %(Tend,np.linalg.norm(uex.values-uend.values,np.inf)/np.linalg.norm(
    #     uex.values,np.inf)))

    # fig = plt.figure(figsize=(8,8))

    # plt.imshow(uend.values[0,:,:])
    # plt.plot(P.state.grid.x.centers,uend.values, color='b', label='SDC')
    # plt.plot(P.state.grid.x.centers,uex.values, color='r', label='Exact')
    # plt.legend()
    # plt.xlim([0, 1])
    # plt.ylim([-1, 1])
    plt.show()
