
from pySDC import CollocationClasses as collclass

from ProblemClass import fenics_heat2d
from fenics_mesh import fenics_mesh, rhs_fenics_mesh
from TransferClass import mesh_to_mesh_fenics
from pySDC.sweeper_classes.mass_matrix_imex import mass_matrix_imex
import pySDC.Methods as mp
from pySDC import Log
from pySDC.Stats import grep_stats, sort_stats

import dolfin as df

if __name__ == "__main__":

    # set global logger (remove this if you do not want the output at all)
    logger = Log.setup_custom_logger('root')

    # nvars = [[8,8],[16,16],[32,32],[64,64],[128,128],[256,256]]
    # dt = [0.125]

    num_procs = 1

    # This comes as read-in for the level class
    lparams = {}
    lparams['restol'] = 8E-11

    sparams = {}
    sparams['maxiter'] = 10

    # This comes as read-in for the problem class
    pparams = {}
    pparams['nu'] = 0.1
    pparams['t0'] = 0.0 # ugly, but necessary to set up ProblemClass
    pparams['c_nvars'] = [[512]]#,[256]]
    pparams['family'] = 'CG'
    pparams['order'] = [1]
    # pparams['levelnumber'] = [2,1]

    # This comes as read-in for the transfer operations
    tparams = {}
    tparams['finter'] = True

    # Fill description dictionary for easy hierarchy creation
    description = {}
    description['problem_class'] = fenics_heat2d
    description['problem_params'] = pparams
    description['dtype_u'] = fenics_mesh
    description['dtype_f'] = rhs_fenics_mesh
    description['collocation_class'] = collclass.CollGaussLobatto
    description['num_nodes'] = 3
    description['sweeper_class'] = mass_matrix_imex
    description['level_params'] = lparams
    description['transfer_class'] = mesh_to_mesh_fenics
    description['transfer_params'] = tparams

    # quickly generate block of steps
    MS = mp.generate_steps(num_procs,sparams,description)

    # setup parameters "in time"
    t0 = MS[0].levels[0].prob.t0
    dt = 0.5
    Tend = 1*dt

    # get initial values on finest level
    P = MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # Pf = MS[0].levels[0].prob
    # uinit_f = Pf.u_exact(t0)
    #
    # Pg = MS[0].levels[1].prob
    # uinit_g = Pg.u_exact(t0)
    #
    # urest = df.project(uinit_f.values,Pg.V)
    # uinter = df.project(uinit_g.values,Pf.V)
    #
    # print(df.norm(urest.vector() - uinit_g.values.vector()))
    # print(df.norm(uinter.vector() - uinit_f.values.vector()))

    # u = df.TrialFunction(Pg.V)
    # v = df.TestFunction(Pg.V)
    # lhs = u*v*df.dx
    # A = df.assemble(lhs)
    # rhs = uinit_f.values*v*df.dx
    # b = df.assemble(rhs) ## This line gives an error because v is an "Argument"
    # urest = fenics_mesh(uinit_g)
    # df.solve(A, urest.values.vector(), rhs)

    # meshc = df.UnitSquareMesh(4,4)
    # meshf = df.refine(meshc)
    # Vc = df.FunctionSpace(meshc,'CG',1)
    # Vf = df.FunctionSpace(meshf,'CG',1)
    # f = df.Expression("sin(2.0*pi*x[0])")
    # f_fine = df.project(f, Vf)
    # f_coarse = df.project(f, Vc)
    #
    # # Approach #1 you suggested
    # u = df.TrialFunction(Vc)
    # v = df.TestFunction(Vc)
    # lhs = u*v*df.dx
    # A = df.assemble(lhs)
    # rhs = f_fine*v*df.dx(meshc)
    # b = df.assemble(rhs)
    # f_l2 = df.Function(Vc)
    # df.solve(A, f_l2.vector(), b)
    # print(len(f_l2.vector().array()),len(f_coarse.vector().array()))
    # print(df.norm(f_l2.vector()-f_coarse.vector(),'linf'))
    #
    # frest = df.project(f_fine,Vc)
    # print(len(frest.vector().array()),len(f_coarse.vector().array()))
    # print(df.norm(frest.vector()-f_coarse.vector(),'linf'))
    #
    # # Approach #1 you suggested
    # u = df.TrialFunction(Vf)
    # v = df.TestFunction(Vf)
    # lhs = u*v*df.dx
    # A = df.assemble(lhs)
    # rhs = f_coarse*v*df.dx(meshf)
    # b = df.assemble(rhs)
    # f_l2 = df.Function(Vf)
    # # df.solve(A, f_l2.vector(), b)
    # df.solve(lhs==rhs,f_l2)
    # print(len(f_l2.vector().array()),len(f_fine.vector().array()))
    # print(df.norm(f_l2.vector()-f_fine.vector(),'linf'))
    #
    # finter = df.project(f_coarse,Vf)
    # print(len(finter.vector().array()),len(f_fine.vector().array()))
    # print(df.norm(finter.vector()-f_fine.vector(),'linf'))
    #
    # exit()
    #
    # urest = fenics_mesh(uinit_g)
    # urest.values = df.project(uinit_f.values,Pg.V)
    #
    # uinter = fenics_mesh(uinit_f)
    # uinter.values = df.project(uinit_g.values,Pf.V)
    #
    # print(len(uinit_g.values.vector().array()),len(urest.values.vector().array()))
    # print(abs(uinit_g-urest))
    # print(len(uinit_f.values.vector().array()),len(uinter.values.vector().array()))
    # print(abs(uinit_f-uinter))
    #
    # exit()


    # call main function to get things done...
    uend,stats = mp.run_pfasst_serial(MS,u0=uinit,t0=t0,dt=dt,Tend=Tend)

    # df.plot(uend.values,interactive=True)

    # compute exact solution and compare
    uex = P.u_exact(Tend)

    print('error at time %s: %s' %(Tend,abs(uex-uend)/abs(uex)))

    extract_stats = grep_stats(stats,iter=-1,type='residual')
    sortedlist_stats = sort_stats(extract_stats,sortby='step')
    print(extract_stats,sortedlist_stats)