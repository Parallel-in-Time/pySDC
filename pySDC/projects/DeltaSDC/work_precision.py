"""
Work-precision with genuinely low-precision solves: for a target error at T, the cheapest (M, dt, restol) in fp64
and with formats chosen live from the previous step (first step fp64). Work = node solves weighted by format cost.

Usage: ``python -m pySDC.projects.DeltaSDC.work_precision <heat|advection|allencahn> <M>`` for one sweep
(writes ``data/wp_<problem>_M<M>.pkl``; run all twelve in parallel), then
``python -m pySDC.projects.DeltaSDC.work_precision figure`` for the fronts and the table of the note.
"""

import pickle
import sys

import numpy as np

from pySDC.implementations.problem_classes.AdvectionEquation_ND_FD import advectionNd
from pySDC.projects.DeltaSDC.eta_spec import ADVECTION, ALLEN_CAHN, HEAT, case
from pySDC.projects.DeltaSDC.genuine import COST, genuine
from pySDC.projects.DeltaSDC.inexact import FORMATS
from pySDC.projects.DeltaSDC.precision_theory import bound, kappa, record_run, schedule
from pySDC.projects.DeltaSDC.problems import allencahn_delta, heat_delta

LINEAR = {'linear_implicit': True}
PROBLEMS = {
    'heat': dict(cls=heat_delta, params=dict(HEAT, freq=1), T=0.1, dts=[0.1 / 2**j for j in range(8)], sweeper=LINEAR),
    'advection': dict(
        cls=advectionNd, params=ADVECTION, T=1.0, dts=[s / 128 for s in (16, 8, 4, 2, 1, 0.5, 0.25)], sweeper=LINEAR
    ),
    'allencahn': dict(
        cls=allencahn_delta, params=ALLEN_CAHN, T=0.016, dts=[0.016 / 2**j for j in range(0, 7)], sweeper=None
    ),
}
RESTOLS = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
PRECISION = {'fp16': 'float16', 'fp32': 'float32', 'fp64': None}


def spec_for(name, dt, restol, M):
    p = PROBLEMS[name]
    return case(name, name, p['cls'], p['params'], dt, restol, M=M, sweeper=p['sweeper'])


def trajectory(name, dt, restol, M, mixed):
    """Integrate to T; the formats of every step after the first come from its predecessor's run."""
    p = PROBLEMS[name]
    spec = spec_for(name, dt, restol, M)
    nonlinear = p['cls'] is allencahn_delta
    u, previous, work, used = None, None, 0.0, []
    # first step: fp32 if the a-priori stiffness check passes with a wide margin, else fp64
    formats = ['fp32'] if mixed and kappa(spec) * FORMATS['fp32'] <= 1e-3 else ['fp64']
    for n in range(int(round(p['T'] / dt))):
        if mixed and previous is not None and len(previous['history']) >= 2:
            if nonlinear:  # the library's Newton-Krylov solve takes one format per step
                eta = bound(spec, run=previous)['eta_star']
                formats = [next((f for f in ('fp16', 'fp32') if FORMATS[f] <= eta), 'fp64')]
            else:
                formats = schedule(spec, run=previous)
        if nonlinear:
            params = dict(spec['problem_params'], solve_precision=PRECISION[formats[0]])
            run = record_run(spec, problem_params=params, u0=u, t0=n * dt, restol=restol, maxiter=50)
        else:
            params = dict(spec['problem_params'], schedule=tuple(formats), nodes=M)
            run = record_run(
                spec, problem_class=genuine(p['cls']), problem_params=params, u0=u, t0=n * dt, restol=restol, maxiter=50
            )
        its = len(run['history'])
        work += M * sum(COST[formats[min(k, len(formats) - 1)]] for k in range(its))
        used.append(formats[:its] if not nonlinear else formats * its)
        u, previous = run['uend'], run
    return np.asarray(u).copy(), work, used


def reference(name):
    p = PROBLEMS[name]
    dt = p['dts'][-1] / 4
    spec = spec_for(name, dt, 1e-13, 5)
    u = None
    for n in range(int(round(p['T'] / dt))):
        u = record_run(spec, u0=u, t0=n * dt, restol=1e-13, maxiter=50)['uend']
    return np.asarray(u).copy()


NAMES = {'heat': 'heat', 'advection': 'advection', 'allencahn': 'Allen--Cahn'}
TARGETS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]


def load(name):
    """All sweeps of one problem."""
    import glob

    return [r for f in sorted(glob.glob(f'data/wp_{name}_M*.pkl')) for r in pickle.load(open(f, 'rb'))]


def best(rows, mixed, target):
    """The cheapest configuration of a kind that reaches the target error, or None."""
    ok = [r for r in rows if r['mixed'] == mixed and r['err'] <= target]
    return min(ok, key=lambda r: r['work']) if ok else None


def shares(r):
    """The share of each format over all solves of a configuration."""
    import collections

    c = collections.Counter(f for step in r['used'] for f in step)
    return {k: v / sum(c.values()) for k, v in c.items()}


def front(rows, mixed):
    """The cheapest configuration for every error reached: (work, error) pairs."""
    pts, out, e_best = sorted((r['work'], r['err']) for r in rows if r['mixed'] == mixed), [], np.inf
    for w, e in pts:
        if e < e_best:
            e_best = e
            out.append((w, e))
    return out


def figure():
    """The fronts (data/work_precision.pdf) and the rows of the note's table (data/tab_wp.tex)."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lines = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, name in zip(axes, NAMES):
        rows = load(name)
        print(name, len(rows), 'runs')
        for t in TARGETS:
            a, b = best(rows, False, t), best(rows, True, t)
            if not a or not b:
                continue
            sh = shares(b)
            fm = ', '.join(f"{k} {100 * sh[k]:.0f}\\%" for k in ('fp16', 'fp32', 'fp64') if sh.get(k, 0) >= 0.005)
            print(
                f"  {t:.0e}: fp64 M={a['M']} dt={a['dt']:.2e} restol={a['restol']:.0e} W={a['work']:.1f} | mixed M={b['M']} dt={b['dt']:.2e} restol={b['restol']:.0e} W={b['work']:.1f} {fm} | gain {a['work'] / b['work']:.2f}"
            )
            tt = f'$10^{{{int(np.log10(t))}}}$'
            lines.append(
                f"{NAMES[name]} & {tt} & {a['M']} & {a['dt']:.1e} & {a['work']:.0f} & {b['M']} & {b['dt']:.1e} & {b['work']:.0f} & {fm} & {a['work'] / b['work']:.1f}"
                + r' \\'
            )
        lines.append(r'\midrule')
        for mixed, color, label in ((False, '#6b6b6b', 'fp64'), (True, '#2a78d6', 'formats from the previous step')):
            pts = [(r['work'], r['err']) for r in rows if r['mixed'] == mixed]
            ax.loglog(*zip(*pts), 'o', color=color, alpha=0.15, markersize=3)
            ax.loglog(*zip(*front(rows, mixed)), '-o', color=color, label=label, markersize=4, drawstyle='steps-post')
        ax.set_xlabel('work (node solves, weighted by format size)')
        ax.set_ylabel('relative error at $T$')
        ax.set_title(NAMES[name].replace('--', '-'))
        ax.set_ylim(1e-12, 1e-1)
        ax.grid(True, alpha=0.4, linewidth=0.6)
        ax.legend(frameon=False, loc='lower left')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig('data/work_precision.pdf')
    fig.savefig('data/work_precision.png', dpi=150)
    open('data/tab_wp.tex', 'w').write('\n'.join(lines[:-1]) + '\n')


if __name__ == '__main__':
    if sys.argv[1] == 'figure':
        figure()
        sys.exit()
    name, M = sys.argv[1], int(sys.argv[2])
    try:
        ref = pickle.load(open(f'data/ref_{name}.pkl', 'rb'))
    except FileNotFoundError:
        ref = reference(name)
        pickle.dump(ref, open(f'data/ref_{name}.pkl', 'wb'))
    rows = []
    for dt in PROBLEMS[name]['dts']:
        for restol in RESTOLS:
            for mixed in (False, True):
                try:
                    u, work, used = trajectory(name, dt, restol, M, mixed)
                    err = np.abs(u - ref).max() / np.abs(ref).max()
                except Exception as e:  # a configuration that fails (e.g. Newton) is reported, not fatal
                    print(
                        f'{name} M={M} dt={dt:.2e} restol={restol:.0e} {"mixed" if mixed else "fp64 "} failed: {e!r}',
                        flush=True,
                    )
                    continue
                rows.append(dict(M=M, dt=dt, restol=restol, mixed=mixed, err=err, work=work, used=used))
                print(
                    f'{name} M={M} dt={dt:.2e} restol={restol:.0e} {"mixed" if mixed else "fp64 "} err={err:.2e} work={work:8.1f}',
                    flush=True,
                )
    pickle.dump(rows, open(f'data/wp_{name}_M{M}.pkl', 'wb'))
