import numpy as np

from projects.FastWaveSlowWave.HookClass_boussinesq import gmres_tolerance
from pySDC.implementations.collocation_classes.gauss_legendre import CollGaussLegendre
from pySDC.implementations.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.implementations.problem_classes.Boussinesq_2D_FD_imex import boussinesq_2d_imex
from pySDC.implementations.problem_classes.boussinesq_helpers.unflatten import unflatten
from pySDC.implementations.problem_classes.boussinesq_helpers.standard_integrators import SplitExplicit, dirk, rk_imex
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI


def main():
    """

    """

    num_procs = 1

    # setup parameters "in time"
    t0 = 0
    Tend = 3000
    Nsteps = 100
    dt = Tend / float(Nsteps)

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-15
    level_params['dt'] = dt

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 4

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussLegendre
    sweeper_params['num_nodes'] = 3

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20

    # initialize problem parameters
    problem_params = dict()
    problem_params['nvars'] = [(4, 300, 30)]
    problem_params['u_adv'] = 0.02
    problem_params['c_s'] = 0.3
    problem_params['Nfreq'] = 0.01
    problem_params['x_bounds'] = [(-150.0, 150.0)]
    problem_params['z_bounds'] = [(0.0, 10.0)]
    problem_params['order'] = [4]
    problem_params['order_upw'] = [5]
    problem_params['gmres_maxiter'] = [500]
    problem_params['gmres_restart'] = [10]
    problem_params['gmres_tol_limit'] = [1e-05]
    problem_params['gmres_tol_factor'] = [0.1]

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = boussinesq_2d_imex  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['dtype_u'] = mesh  # pass data type for u
    description['dtype_f'] = rhs_imex_mesh  # pass data type for f
    description['sweeper_class'] = imex_1st_order  # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['hook_class'] = gmres_tolerance

    # ORDER OF DIRK/IMEX EQUAL TO NUMBER OF SDC ITERATIONS AND THUS SDC ORDER
    dirk_order = step_params['maxiter']

    controller = allinclusive_classic_nonMPI(num_procs=num_procs, controller_params=controller_params,
                                             description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    cfl_advection = P.params.u_adv * dt / P.h[0]
    cfl_acoustic_hor = P.params.c_s * dt / P.h[0]
    cfl_acoustic_ver = P.params.c_s * dt / P.h[1]
    print("Horizontal resolution: %4.2f" % P.h[0])
    print("Vertical resolution:   %4.2f" % P.h[1])
    print("CFL number of advection: %4.2f" % cfl_advection)
    print("CFL number of acoustics (horizontal): %4.2f" % cfl_acoustic_hor)
    print("CFL number of acoustics (vertical):   %4.2f" % cfl_acoustic_ver)

    print("Running SplitExplicit ....")
    method_split = 'MIS4_4'
    #   method_split = 'RK3'
    splitp = SplitExplicit(P, method_split, problem_params)
    u0 = uinit.values.flatten()
    usplit = np.copy(u0)
    print(np.linalg.norm(usplit))
    for i in range(0, 2 * Nsteps):
        usplit = splitp.timestep(usplit, dt / 2)
    print(np.linalg.norm(usplit))

    print("Running DIRK ....")
    dirkp = dirk(P, dirk_order)
    udirk = np.copy(u0)
    print(np.linalg.norm(udirk))
    for i in range(0, Nsteps):
        udirk = dirkp.timestep(udirk, dt)
    print(np.linalg.norm(udirk))

    print("Running RK-IMEX ....")
    rkimex = rk_imex(P, dirk_order)
    uimex = np.copy(u0)
    dt_imex = dt
    for i in range(0, Nsteps):
        uimex = rkimex.timestep(uimex, dt_imex)
    print(np.linalg.norm(uimex))

    print("Running SDC...")
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # For reference solution, increase GMRES tolerance
    P.gmres_tol_limit = 1e-10
    rkimexref = rk_imex(P, 5)
    uref = np.copy(u0)
    dt_ref = dt / 10.0
    print("Running RK-IMEX reference....")
    for i in range(0, 10 * Nsteps):
        uref = rkimexref.timestep(uref, dt_ref)

    usplit = unflatten(usplit, 4, P.N[0], P.N[1])
    udirk = unflatten(udirk, 4, P.N[0], P.N[1])
    uimex = unflatten(uimex, 4, P.N[0], P.N[1])
    uref = unflatten(uref, 4, P.N[0], P.N[1])

    np.save('xaxis', P.xx)
    np.save('sdc', uend.values)
    np.save('dirk', udirk)
    np.save('rkimex', uimex)
    np.save('split', usplit)
    np.save('uref', uref)

    print("diff split  ", np.linalg.norm(uref - usplit))
    print("diff dirk   ", np.linalg.norm(uref - udirk))
    print("diff rkimex ", np.linalg.norm(uref - uimex))
    print("diff sdc    ", np.linalg.norm(uref - uend.values))

    print(" #### Logging report for Split    #### ")
    print("Total number of matrix multiplications: %5i" % splitp.logger.nsmall)

    print(" #### Logging report for DIRK-%1i #### " % dirkp.order)
    print("Number of calls to implicit solver: %5i" % dirkp.logger.solver_calls)
    print("Total number of GMRES iterations: %5i" % dirkp.logger.iterations)
    print("Average number of iterations per call: %6.3f" %
          (float(dirkp.logger.iterations) / float(dirkp.logger.solver_calls)))
    print(" ")
    print(" #### Logging report for RK-IMEX-%1i #### " % rkimex.order)
    print("Number of calls to implicit solver: %5i" % rkimex.logger.solver_calls)
    print("Total number of GMRES iterations: %5i" % rkimex.logger.iterations)
    print("Average number of iterations per call: %6.3f" %
          (float(rkimex.logger.iterations) / float(rkimex.logger.solver_calls)))
    print(" ")
    print(" #### Logging report for SDC-(%1i,%1i) #### " % (sweeper_params['num_nodes'], step_params['maxiter']))
    print("Number of calls to implicit solver: %5i" % P.gmres_logger.solver_calls)
    print("Total number of GMRES iterations: %5i" % P.gmres_logger.iterations)
    print("Average number of iterations per call: %6.3f" %
          (float(P.gmres_logger.iterations) / float(P.gmres_logger.solver_calls)))


if __name__ == "__main__":
    main()
