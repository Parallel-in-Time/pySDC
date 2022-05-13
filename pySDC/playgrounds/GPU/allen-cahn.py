from pySDC.implementations.problem_classes.AllenCahn_2D_FD import allencahn_semiimplicit
# from pySDC.implementations.problem_classes.AllenCahn_2D_FD_gpu import allencahn_semiimplicit
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.helpers.stats_helper import filter_stats, sort_stats
# from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh
import numpy as np
import pickle
import matplotlib.pylab as plt

Ns = np.asarray([128, 256, 512, 1024, 2048])
# initialize problem parameters
problem_params = dict()
problem_params['nu'] = 2
problem_params['eps'] = 0.04
problem_params['radius'] = 0.25
# problem_params['nvars'] = [(1024, 1024)]
# problem_params['newton_maxiter'] = 100
# problem_params['newton_tol'] = 1E-08
problem_params['lin_tol'] = 1E-10
problem_params['lin_maxiter'] = 99
# problem_params['dtype_u'] = cupy_mesh
# problem_params['dtype_f'] = cupy_mesh

# initialize level parameters
level_params = dict()
level_params['restol'] = 1E-07
level_params['dt'] = 1E-03
level_params['nsweeps'] = 1

# initialize sweeper parameters
sweeper_params = dict()
sweeper_params['collocation_class'] = CollGaussRadau_Right
sweeper_params['QI'] = ['LU']
sweeper_params['QE'] = ['PIC']
sweeper_params['num_nodes'] = 3
sweeper_params['initial_guess'] = 'spread'

# initialize step parameters
step_params = dict()
step_params['maxiter'] = 50

# setup parameters "in time"
t0 = 0
Tend = 8*1E-03

# initialize controller parameters
controller_params = dict()
controller_params['logger_level'] = 10
for N in Ns:
    problem_params['nvars'] = [(N, N)]
    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = allencahn_semiimplicit
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # write down stats to .pickle file
    # name = f'pickle/ac-jusuf-pySDC-N{N}-gpu.pickle'
    name = f'pickle/ac-jusuf-pySDC-N{N}-cpu.pickle'
    data = {
        'P': P,
        'nvars': problem_params['nvars'],
        'stats': stats
    }
    with open(name, 'wb') as f:
        pickle.dump(data, f)


