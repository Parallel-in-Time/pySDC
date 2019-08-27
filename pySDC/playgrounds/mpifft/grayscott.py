import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.collocation_classes.gauss_lobatto import CollGaussLobatto
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.sweeper_classes.multi_implicit import multi_implicit
from pySDC.implementations.problem_classes.GrayScott_MPIFFT import grayscott_imex_diffusion, grayscott_imex_linear, \
    grayscott_mi_diffusion, grayscott_mi_linear
from pySDC.implementations.transfer_classes.TransferMesh_MPIFFT import fft_to_fft


def run_simulation(spectral=None, splitting_type=None, ml=None, num_procs=None):
    """
    A test program to do SDC, MLSDC and PFASST runs for the 2D NLS equation

    Args:
        spectral (bool): run in real or spectral space
        ml (bool): single or multiple levels
        num_procs (int): number of parallel processors
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-12
    level_params['dt'] = 8E-00
    level_params['nsweeps'] = [1]
    level_params['residual_type'] = 'last_abs'

    # initialize sweeper parameters
    sweeper_params = dict()
    # sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['collocation_class'] = CollGaussLobatto
    sweeper_params['num_nodes'] = [5]
    sweeper_params['QI'] = ['LU']  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['Q1'] = ['LU']  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['Q2'] = ['LU']  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['QE'] = ['EE']  # You can try PIC here, but PFASST doesn't like this..
    sweeper_params['initial_guess'] = 'spread'

    # initialize problem parameters
    problem_params = dict()
    if ml:
        problem_params['nvars'] = [(128, 128), (32, 32)]
    else:
        problem_params['nvars'] = [(128, 128)]
    problem_params['spectral'] = spectral
    problem_params['comm'] = comm
    problem_params['Du'] = 0.00002
    problem_params['Dv'] = 0.00001
    problem_params['A'] = 0.04
    problem_params['B'] = 0.1
    problem_params['newton_maxiter'] = 50
    problem_params['newton_tol'] = 1E-11

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 100
    step_params['errtol'] = 1E-09

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20 if rank == 0 else 99
    # controller_params['predict_type'] = 'fine_only'
    controller_params['use_iteration_estimator'] = False

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_params'] = problem_params  # pass problem parameters
    if splitting_type == 'diffusion':
        description['problem_class'] = grayscott_imex_diffusion
    elif splitting_type == 'linear':
        description['problem_class'] = grayscott_imex_linear
    elif splitting_type == 'mi_diffusion':
        description['problem_class'] = grayscott_mi_diffusion
    elif splitting_type == 'mi_linear':
        description['problem_class'] = grayscott_mi_linear
    else:
        raise NotImplementedError(f'splitting_type = {splitting_type} not implemented')
    if splitting_type == 'mi_diffusion' or splitting_type == 'mi_linear':
        description['sweeper_class'] = multi_implicit
    else:
        description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = fft_to_fft

    # set time parameters
    t0 = 0.0
    Tend = 32

    f = None
    if rank == 0:
        f = open('GS_out.txt', 'a')
        out = f'Running with ml = {ml} and num_procs = {num_procs}...'
        f.write(out + '\n')
        print(out)

    # instantiate controller
    controller = controller_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # plt.figure()
    # plt.imshow(uinit[..., 0], vmin=0, vmax=1)
    # plt.title('v')
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(uinit[..., 1], vmin=0, vmax=1)
    # plt.title('v')
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(uinit[..., 0] + uinit[..., 1])
    # plt.title('sum')
    # plt.colorbar()

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # plt.figure()
    # plt.imshow(P.fft.backward(uend[..., 0]))#, vmin=0, vmax=1)
    # # plt.imshow(np.fft.irfft2(uend.values[..., 0]))#, vmin=0, vmax=1)
    # plt.title('u')
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(P.fft.backward(uend[..., 1]))#, vmin=0, vmax=1)
    # # plt.imshow(np.fft.irfft2(uend.values[..., 1]))#, vmin=0, vmax=1)
    # plt.title('v')
    # plt.colorbar()
    # # plt.figure()
    # # plt.imshow(uend[..., 0] + uend[..., 1])
    # # plt.title('sum')
    # # plt.colorbar()
    # plt.show()
    # # exit()

    if rank == 0:
        # filter statistics by type (number of iterations)
        filtered_stats = filter_stats(stats, type='niter')

        # convert filtered statistics to list of iterations count, sorted by process
        iter_counts = sort_stats(filtered_stats, sortby='time')

        niters = np.array([item[1] for item in iter_counts])
        out = f'   Min/Mean/Max number of iterations: ' \
              f'{np.min(niters):4.2f} / {np.mean(niters):4.2f} / {np.max(niters):4.2f}'
        f.write(out + '\n')
        print(out)
        out = '   Range of values for number of iterations: %2i ' % np.ptp(niters)
        f.write(out + '\n')
        print(out)
        out = '   Position of max/min number of iterations: %2i -- %2i' % \
              (int(np.argmax(niters)), int(np.argmin(niters)))
        f.write(out + '\n')
        print(out)
        out = '   Std and var for number of iterations: %4.2f -- %4.2f' % (float(np.std(niters)), float(np.var(niters)))
        f.write(out + '\n')
        print(out)

        timing = sort_stats(filter_stats(stats, type='timing_run'), sortby='time')
        out = f'Time to solution: {timing[0][1]:6.4f} sec.'
        f.write(out + '\n')
        print(out)

        f.write('\n')
        print()
        f.close()


def main():
    """
    Little helper routine to run the whole thing

    Note: This can also be run with "mpirun -np 2 python grayscott.py"
    """
    # run_simulation(spectral=False, splitting_type='diffusion', ml=False, num_procs=1)
    # run_simulation(spectral=True, splitting_type='diffusion', ml=False, num_procs=1)
    # run_simulation(spectral=True, splitting_type='linear', ml=False, num_procs=1)
    # run_simulation(spectral=False, splitting_type='diffusion', ml=True, num_procs=1)
    # run_simulation(spectral=True, splitting_type='diffusion', ml=True, num_procs=1)
    # run_simulation(spectral=False, splitting_type='diffusion', ml=True, num_procs=10)
    # run_simulation(spectral=True, splitting_type='diffusion', ml=True, num_procs=10)

    # run_simulation(spectral=False, splitting_type='mi_diffusion', ml=False, num_procs=1)
    run_simulation(spectral=True, splitting_type='mi_diffusion', ml=False, num_procs=1)
    # run_simulation(spectral=False, splitting_type='mi_linear', ml=False, num_procs=1)
    # run_simulation(spectral=True, splitting_type='mi_linear', ml=False, num_procs=1)


if __name__ == "__main__":
    main()
