import numpy as np
import copy as cp

from pySDC.implementations.problem_classes.HeatEquation_1D_FD_forced import heat1d_forced
from pySDC.implementations.problem_classes.HeatEquation_1D_FD_periodic import heat1d_periodic
from pySDC.implementations.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh

from pySDC.implementations.controller_classes.allinclusive_multigrid_nonMPI import allinclusive_multigrid_nonMPI


from pySDC.core import Step as stepclass

from joblib import Parallel, delayed
import time
import pickle
from mpi4py import MPI

import pathos.pools as pp

import dill as pickle

import cProfile


def test_function(description):
    return stepclass.step(description)


def main():
    """
    A simple test program to do PFASST runs for the heat equation
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-09
    level_params['dt'] = 0.25

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part

    # initialize problem parameters
    problem_params = dict()
    problem_params['nu'] = 0.1  # diffusion coefficient
    problem_params['freq'] = 2  # frequency for the test value
    problem_params['nvars'] = [8191, 4095]  # number of degrees of freedom for each level

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 2
    space_transfer_params['periodic'] = False

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 10

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = heat1d_forced                  # pass problem class
    description['problem_params'] = problem_params                # pass problem parameters
    description['dtype_u'] = mesh                                 # pass data type for u
    description['dtype_f'] = rhs_imex_mesh                        # pass data type for f
    description['sweeper_class'] = imex_1st_order                 # pass sweeper
    description['sweeper_params'] = sweeper_params                # pass sweeper parameters
    description['level_params'] = level_params                    # pass level parameters
    description['step_params'] = step_params                      # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh            # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

    # set time parameters
    t0 = 0.0
    Tend = 1.0

    num_steps = 4

    # startt = time.time()
    #
    # p = pp.ProcessPool(num_procs)
    # MSp = p.map(test_function, num_steps * [description])
    #
    # # MSp = Parallel(n_jobs=2, backend="threading")(map(delayed(test_function), num_procs * [description]))
    #
    # # S = stepclass.step(description)
    # #
    # endt = time.time()
    # # endt_all = comm.allreduce(endt-startt, op=MPI.MAX)
    #
    # # print(num_procs, comm.Get_rank(), endt_all)
    # print(endt-startt)
    #
    # #
    # print(MSp)
    # #
    # startt = time.time()
    # MS = []
    # for p in range(num_steps):
    #     MS.append(stepclass.step(description))
    # endt = time.time()
    # print(endt-startt)
    #
    # # MS.append(cp.deepcopy(MS[0]))
    #
    # pickle.dump(MS[0], open("save.p", "wb"))
    # # print(endt-startt)
    #
    # print(MS)

    # instantiate controller

    controller = allinclusive_multigrid_nonMPI(num_procs=num_steps, controller_params=controller_params,
                                               description=description)


    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)


    startt = time.time()
    # pr = cProfile.Profile()
    # pr.enable()

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # pr.disable()
    endt = time.time()
    print(endt-startt)

    # pr.print_stats(sort=2)


if __name__ == "__main__":
    main()