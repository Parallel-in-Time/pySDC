from pySDC.implementations.problem_classes.HeatEquation_ND_FD_forced_periodic import heatNd_periodic
# from pySDC.implementations.problem_classes.HeatEquation_ND_FD_forced_periodic_gpu import heatNd_periodic
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.helpers.stats_helper import filter_stats, sort_stats
# from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh
import numpy as np
import pickle
import matplotlib.pylab as plt
name = 'pickle/heat-jusuf-pySDC-cpu_f.pickle'
# name = 'pickle/heat-jusuf-pySDC-gpu_f.pickle'
Ns = np.asarray([16, 32, 64, 128, 256])
D = 3
# Ns = np.asarray([128, 256, 512])
times = np.zeros_like(Ns, dtype=float)
setup = np.zeros_like(Ns, dtype=float)
cg = np.zeros_like(Ns, dtype=float)
cg_Count = np.zeros_like(Ns)
f_im = np.zeros_like(Ns, dtype=float)
f_ex = np.zeros_like(Ns, dtype=float)
# initialize problem parameters
problem_params = dict()
problem_params['nu'] = 1
problem_params['freq'] = [4, 4, 4]
problem_params['order'] = 2
problem_params['ndim'] = D
problem_params['lintol'] = 1E-10
problem_params['liniter'] = 99
problem_params['direct_solver'] = True

# initialize level parameters
level_params = dict()
level_params['restol'] = 1E-07
level_params['dt'] = 1E-07
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
schritte = 8
Tend = schritte*level_params['dt']

# initialize controller parameters
controller_params = dict()
controller_params['logger_level'] = 30
for i, N in enumerate(Ns):
    problem_params['nvars'] = [(N, N, N)]
    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = heatNd_periodic
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

    # get the stats
    timing = sort_stats(filter_stats(stats, type='timing_run'), sortby='time')
    times[i] = timing[0][1]
    timing_setup = sort_stats(filter_stats(stats, type='timing_setup'), sortby='time')
    setup[i] = timing_setup[0][1]
    timing_step = sort_stats(filter_stats(stats, type='timing_step'), sortby='time')
    timing_step = [ts[1] for ts in timing_step]
    cg[i] = np.asarray(sum(timing_step))
    cg_Count[i] = P.lin_ncalls
    f_im[i] = P.f_im
    f_ex[i] = P.f_ex
# write down stats to .pickle file
data = {
    'Ns': Ns,
    'D': D,
    'dt': level_params['dt'],
    'schritte': schritte,
    'iteration': step_params['maxiter'],
    'Tolerance': problem_params['lin_tol'],
    'times': times,
    'setup': setup,
    'cg-time': cg,
    'cg-count': cg_Count,
    'f-time-imp': f_im,
    'f-time-exp': f_ex

}
with open(name, 'wb') as f:
    pickle.dump(data, f)


