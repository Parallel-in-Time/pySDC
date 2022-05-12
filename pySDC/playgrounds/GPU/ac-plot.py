import pickle
import numpy as np
from pySDC.helpers.stats_helper import filter_stats, sort_stats

name = 'pickle/ac-jusuf-pySDC.pickle'
with open(name, 'rb') as f:
   data = pickle.load(f)
nvars = data['nvars']
stats = data['stats']
P = data['P']
# filter statistics by variant (number of iterations)
filtered_stats = filter_stats(stats, type='niter')

# convert filtered statistics to list of iterations count, sorted by time-step
iter_counts = sort_stats(filtered_stats, sortby='time')

# compute and print statistics
niters = np.array([item[1] for item in iter_counts])
print('CPU NxN: ', nvars)
print(f'Mean number of iterations: {np.mean(niters):4.2f}')
print(f'Range of values for number of iterations: {np.ptp(niters)}')
print(f'Position of max/min number of iterations: {int(np.argmax(niters))} / {int(np.argmax(niters))}')
print(f'Iteration count nonlinear solver (sum/mean per call): '
      f'{P.newton_itercount} / {P.newton_itercount / max(P.newton_ncalls, 1)}')
timing_step = sort_stats(filter_stats(stats, type='timing_step'), sortby='time')
print('Timing step:\n', timing_step)
timing = sort_stats(filter_stats(stats, type='timing_run'), sortby='time')
print('Time to solution: %6.4f sec.' % timing[0][1])


