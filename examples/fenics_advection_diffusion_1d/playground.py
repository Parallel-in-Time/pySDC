import pySDC.deprecated.PFASST_blockwise_old as mp
from examples.fenics_advection_diffusion_1d.ProblemClass import fenics_adv_diff_1d
from examples.fenics_advection_diffusion_1d.TransferClass import mesh_to_mesh_fenics
from pySDC import CollocationClasses as collclass
from pySDC import Log
from pySDC.datatype_classes.fenics_mesh import fenics_mesh,rhs_fenics_mesh
from pySDC.sweeper_classes.imex_1st_order import imex_1st_order

if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    num_procs = 1

    # assert num_procs == 1,'turn on predictor!'

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 5E-09

    sparams = {}
    sparams['maxiter'] = 20

    # This comes as read-in for the problem class
    pparams = {}
    pparams['nu'] = 0.05
    pparams['mu'] = 1.0
    pparams['k'] = 1
    pparams['t0'] = 0.0 # ugly, but necessary to set up ProblemClass
    # pparams['c_nvars'] = [(16,16)]
    pparams['c_nvars'] = [128]
    pparams['family'] = 'CG'
    pparams['order'] = [1]
    pparams['refinements'] = [1,0]


    # This comes as read-in for the transfer operations
    tparams = {}
    tparams['finter'] = True

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = fenics_adv_diff_1d
    description['problem_params'] = pparams
    description['dtype_u'] = fenics_mesh
    description['dtype_f'] = rhs_fenics_mesh
    description['collocation_class'] = collclass.CollGaussLegendre
    description['num_nodes'] = 3
    description['sweeper_class'] = imex_1st_order
    description['level_params'] = lparams
    description['transfer_class'] = mesh_to_mesh_fenics
    description['transfer_params'] = tparams

    # quickly generate block of steps
    MS = mp.generate_steps(num_procs,sparams,description)

    # setup parameters "in time"
    t0 = MS[0].levels[0].prob.t0
    dt = 0.2
    Tend = 1.6

    # get initial values on finest level
    P = MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend,stats = mp.run_pfasst(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

    # df.plot(uend.values,interactive=True)

    # compute exact solution and compare
    uex = P.u_exact(Tend)

    print('(classical) error at time %s: %s' %(Tend,abs(uex-uend)/abs(uex)))

    # df.plot(uex.values,key='u')
    # df.plot(uend.values,key='u')
    # df.plot(uex.values-uend.values)
    # df.interactive()

    # uex = df.Expression('sin(a*x[0]) * cos(t)',a=np.pi,t=Tend)
    # print('(fenics-style) error at time %s: %s' %(Tend,df.errornorm(uex,uend.values)))

    # extract_stats = grep_stats(stats,iter=-1,type='residual')
    # sortedlist_stats = sort_stats(extract_stats,sortby='step')
    # print(extract_stats,sortedlist_stats)