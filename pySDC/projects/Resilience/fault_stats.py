import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
import types

import pySDC.helpers.plot_helper as plot_helper
from pySDC.helpers.stats_helper import get_sorted

from pySDC.projects.Resilience.advection import run_advection
from pySDC.projects.Resilience.vdp import run_vdp
from pySDC.projects.Resilience.piline import run_piline

plot_helper.setup_mpl(reset=True)
cmap = TABLEAU_COLORS


def generate_stats(strategy='adaptivity', runs=1000, reload=True, faults=True):
    dat = {
        'level': np.zeros(runs),
        'iteration': np.zeros(runs),
        'node': np.zeros(runs),
        'problem_pos': [],
        'bit': np.zeros(runs),
        'error': np.zeros(runs),
        'total_iteration': np.zeros(runs),
        'restarts': np.zeros(runs),
        'target': np.zeros(runs),
    }

    if reload:
        already_completed = load(strategy, faults)
        if already_completed['runs'] > 0 and already_completed['runs'] <= runs:
            for k in dat.keys():
                dat[k][:min([already_completed['runs'], runs])] = already_completed.get(k, [])
    else:
        already_completed = {'runs': 0}

    if already_completed['runs'] < runs:
        if faults:
            print(f'Doing {strategy} with faults from {already_completed["runs"]} to {runs}')
        else:
            print(f'Doing {strategy} from {already_completed["runs"]} to {runs}')

    for i in range(already_completed['runs'], runs):
        rng = np.random.RandomState(i)
        stats, controller, Tend = prob(strategy, rng, faults)

        faults_run = get_sorted(stats, type='bitflip')

        t, u = get_sorted(stats, type='u', recomputed=False)[-1]

        if t < Tend:
            error = np.inf
        else:
            error = abs(u - controller.MS[0].levels[0].prob.u_exact(t=t))
        total_iteration = sum([k[1] for k in get_sorted(stats, type='k')])

        if faults:
            dat['level'][i] = faults_run[0][1][0]
            dat['iteration'][i] = faults_run[0][1][1]
            dat['node'][i] = faults_run[0][1][2]
            dat['problem_pos'] += [faults_run[0][1][3]]
            dat['bit'][i] = faults_run[0][1][4]
            dat['target'][i] = faults_run[0][1][5]
        dat['error'][i] = error
        dat['total_iteration'][i] = total_iteration
        dat['restarts'][i] = sum([me[1] for me in get_sorted(stats, type='restart')])

    dat['runs'] = runs
    if already_completed['runs'] < runs:
        store(strategy, faults, dat)


def scrutinize(strategy, run, faults=True):
    rng = np.random.RandomState(run)

    force_params = dict()
    force_params['controller_params'] = {'logger_level': 20}

    stats, controller, Tend = prob(strategy, rng, faults, force_params)

    t, u = get_sorted(stats, type='u')[-1]
    k_tot = sum([me[1] for me in get_sorted(stats, type='k')])

    print(f'e={abs(u - controller.MS[0].levels[0].prob.u_exact(t=t)):.2e}, k_tot={k_tot}')
    print('faults:', get_sorted(stats, type='bitflip'))


def convert_faults(faults):
    time = [faults[i][0] for i in range(len(faults))]
    level = [faults[i][1][0] for i in range(len(faults))]
    iteration = [faults[i][1][1] for i in range(len(faults))]
    node = [faults[i][1][2] for i in range(len(faults))]
    problem_pos = [faults[i][1][3] for i in range(len(faults))]
    bit = [faults[i][1][4] for i in range(len(faults))]
    return time, level, iteration, node, problem_pos, bit


def get_path(strategy, faults):
    return f'data/stats/{get_name(strategy, faults)}.pickle'


def get_name(strategy=None, faults=False):
    if prob == run_advection:
        prob_name = 'advection'
    elif prob == run_vdp:
        prob_name = 'vdp'
    elif prob == run_piline:
        prob_name = 'piline'
    else:
        raise NotImplementedError(f'Name not implemented for problem {prob}')

    if faults:
        fault_name = '-faults'
    else:
        fault_name = ''

    if strategy is not None:
        strategy_name = f'-{strategy}'
    else:
        strategy_name = ''

    return f'{prob_name}{strategy_name}{fault_name}'


def store(strategy, faults, dat):
    with open(get_path(strategy, faults), 'wb') as f:
        pickle.dump(dat, f)


def load(strategy, faults):
    try:
        with open(get_path(strategy, faults), 'rb') as f:
            dat = pickle.load(f)
    except FileNotFoundError:
        return {'runs': 0}
    return dat


def get_recovered(strategy, thresh=1.01):
    fault_free = load(strategy, False)
    with_faults = load(strategy, True)

    assert fault_free['error'].std() / fault_free['error'].mean() < 1e-5

    with_faults['recovered'] = with_faults['error'] < thresh * fault_free['error'].mean()
    store(strategy, True, with_faults)


def get_marker(strategy):
    if strategy == 'adaptivity':
        return '*'
    elif strategy == 'nothing':
        return 'o'
    elif strategy == 'iterate':
        return 'v'
    elif strategy == 'HotRod':
        return '^'


def get_color(strategy):
    keys = list(cmap.keys())
    if strategy == 'adaptivity':
        return cmap[keys[1]]
    elif strategy == 'nothing':
        return cmap[keys[0]]
    elif strategy == 'iterate':
        return cmap[keys[2]]
    elif strategy == 'HotRod':
        return cmap[keys[3]]


def rec_rate(dat, no_faults, thingA, mask):
    if len(dat[thingA][mask]) > 0:
        return len(dat[thingA][mask & dat['recovered']]) / len(dat[thingA][mask])
    else:
        return None


def mean(dat, no_faults, thingA, mask):
    return np.mean(dat[thingA][mask])


def extra_mean(dat, no_faults, thingA, mask):
    return np.mean(dat[thingA][mask]) - np.mean(no_faults[thingA])


def plot_thingA_per_thingB(strategy, thingA, thingB, ax=None, glob_mask=None, recovered=False, op=rec_rate):
    dat = load(strategy, True)
    no_faults = load(strategy, False)

    if glob_mask is None:
        glob_mask = np.ones_like(dat[thingB], dtype=bool)

    admissable_thingB = np.unique(dat[thingB][glob_mask])
    me = np.zeros(len(admissable_thingB))
    me_recovered = np.zeros_like(me)

    for i in range(len(me)):
        mask = (dat[thingB] == admissable_thingB[i]) & glob_mask
        if mask.any():
            me[i] = op(dat, no_faults, thingA, mask)
            me_recovered[i] = op(dat, no_faults, thingA, mask & dat['recovered'])

    if recovered:
        ax.plot(admissable_thingB, me_recovered, label=f'{strategy} (only recovered)', color=get_color(strategy),
                marker=get_marker(strategy), ls='--', linewidth=3)

    ax.plot(admissable_thingB, me, label=strategy, color=get_color(strategy), marker=get_marker(strategy), linewidth=2)

    ax.legend(frameon=False)
    ax.set_xlabel(thingB)
    ax.set_ylabel(thingA)


def plot_things_per_things(strategies, thingA, thingB, recovered, mask=None, op=rec_rate,
                           args=types.MappingProxyType({}), name=None):
    fig, ax = plt.subplots(1, 1)
    for s in strategies:
        plot_thingA_per_thingB(s, thingA=thingA, thingB=thingB, recovered=recovered, ax=ax, glob_mask=mask, op=op)

    [plt.setp(ax, k, v) for k, v in args.items()]

    fig.tight_layout()
    plt.savefig(f'data/{get_name()}-{thingA if name is None else name}_per_{thingB}.pdf', transparent=True)
    plt.close(fig)


def print_stats(strategy):
    dat = load(strategy, True)
    strings = {
        'iteration': 'iter',
        'node': 'nodes',
        'bit': 'bits',
    }
    print(f'Stats for {strategy}')
    for k, v in strings.items():
        me = np.unique(dat[k])
        for i in range(len(me)):
            mask = dat[k] == me[i]
            v = f'{v} {len(dat[k][mask])}'
        print(v)


def get_mask(strategy, key=None, val=None, op='eq', old_mask=None):
    dat = load(strategy, True)
    if None in [key, val]:
        mask = dat['bit'] == dat['bit']
    else:
        if op == 'uneq':
            mask = dat[key] != val
        elif op == 'eq':
            mask = dat[key] == val
        elif op == 'leq':
            mask = dat[key] <= val
        elif op == 'geq':
            mask = dat[key] >= val
        elif op == 'lt':
            mask = dat[key] < val
        elif op == 'gt':
            mask = dat[key] > val
        else:
            raise NotImplementedError

    if old_mask is not None:
        return mask & old_mask
    else:
        return mask


def get_index(mask):
    return np.arange(len(mask))[mask]


prob = run_piline
strategies = ['nothing', 'adaptivity', 'iterate']
faults = [False, True]
reload = True
replot = True
max_runs = 1000
step = 50
thresh = 1. + 1e-3
mask = None

for i in range(step, max_runs + step, step):
    prev_len = load(strategies[0], True)['runs']
    for j in range(len(strategies)):
        for f in faults:
            if f:
                runs = i
            else:
                runs = min([5, i])
            generate_stats(strategy=strategies[j], runs=runs, faults=f, reload=reload)
        get_recovered(strategies[j], thresh=thresh)
    reload = True

    if load(strategies[0], True)['runs'] > prev_len or replot:
        # mask = get_mask(strategies[0], 'node', 0, op='uneq')
        # mask = get_mask(strategies[0], 'error', 1e-8, op='leq', old_mask=mask)

        dat = load(strategies[0], True)
        plot_things_per_things(strategies, 'recovered', 'node', False, op=rec_rate, mask=mask,
                               args={'ylabel': 'recovery rate'})
        plot_things_per_things(strategies, 'recovered', 'iteration', False, op=rec_rate, mask=mask,
                               args={'ylabel': 'recovery rate'})
        plot_things_per_things(strategies, 'recovered', 'bit', False, op=rec_rate, mask=mask,
                               args={'ylabel': 'recovery rate'})
        plot_things_per_things(strategies, 'total_iteration', 'bit', True, op=mean, mask=mask,
                               args={'yscale': 'log', 'ylabel': 'total iterations'})
        plot_things_per_things(strategies, 'total_iteration', 'bit', True, op=extra_mean, mask=mask,
                               args={'yscale': 'linear', 'ylabel': 'extra iterations'}, name='extra_iter')
        plot_things_per_things(strategies, 'error', 'bit', True, op=mean, mask=mask, args={'yscale': 'log'})

        replot = False

# for s in strategies:
#     print_stats(s)
# scrutinize('iterate', 181)
# s = 'iterate'
# mask = get_mask(s, 'bit', 1, 'eq')
# mask = get_mask(s, 'node', 0, 'uneq', mask)
# mask = get_mask(s, 'iteration', 3, 'lt', mask)
# mask = get_mask(s, 'iteration', 1, 'gt', mask)
# mask = get_mask(s, 'recovered', True, 'eq', mask)
# print(get_index(mask))
