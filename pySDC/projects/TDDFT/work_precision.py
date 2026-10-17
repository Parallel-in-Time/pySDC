"""
Work-precision comparison of SDC and the propagators of real-time TDDFT codes on the 1D Kohn-Sham molecule.

``python -m pySDC.projects.TDDFT.work_precision`` runs the small demo set (a few minutes) and writes the figure
``data/tddft_work_precision.png``. ``--full`` runs the set behind the numbers in the README (hours on one core;
``--procs`` runs several at once, results are appended to ``data/tddft_full/runs.json`` and finished runs skipped).

Errors are measured against a tightly converged SDC reference on the same grid: the largest deviation of the dipole
over the pulse, relative to its maximum, and the worst odd harmonic yield (in decades).
"""

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from pySDC.projects.TDDFT.propagators import run_propagator
from pySDC.projects.TDDFT.sdc import run_sdc

FULL = dict(problem={}, Tend=448.0, dt_out=1.0)
DEMO = dict(problem=dict(N=256, L=51.2, x_cap=18.0, eta_cap=1e-2, E0=0.02, ncycles=2), Tend=224.0, dt_out=1.0)
PARALLEL = dict(QI='MIN-SR-S', QE='PIC')


def configs(full):
    """(label, method, dt, options) of every run; the first one is the reference"""
    if not full:
        return [
            ('reference', 'sdc', 0.125, dict(M=5, restol=1e-11)),
            ('sdc_parallel', 'sdc', 0.5, dict(M=4, restol=1e-6, **PARALLEL)),
            ('sdc_parallel', 'sdc', 0.25, dict(M=4, restol=1e-8, **PARALLEL)),
            ('sdc_serial', 'sdc', 0.25, dict(M=3, restol=1e-8)),
            ('cfm4', 'cfm4', 0.2, dict(tol=1e-6)),
            ('cfm4', 'cfm4', 0.1, dict(tol=1e-10)),
            ('strang', 'strang', 0.1, {}),
            ('strang', 'strang', 0.025, {}),
        ]
    cs = [('reference', 'sdc', 0.0625, dict(M=5, restol=1e-13))]
    cs += [('strang', 'strang', dt, {}) for dt in [0.2, 0.1, 0.05, 0.025, 0.0125]]
    cs += [('triple_jump', 'triple_jump', dt, {}) for dt in [0.2, 0.1, 0.05, 0.025]]
    cs += [('aetrs', 'aetrs', dt, {}) for dt in [0.04, 0.02, 0.01]]
    cs += [('cfm4', 'cfm4', dt, dict(tol=1e-6)) for dt in [1.0, 0.5, 0.2, 0.1]]
    cs += [('cfm4', 'cfm4', dt, dict(tol=1e-10)) for dt in [0.2, 0.1, 0.05, 0.025]]
    cs += [('crank_nicolson', 'crank_nicolson', dt, {}) for dt in [0.2, 0.1, 0.05]]
    for label, opts in [
        ('sdc_serial', dict(M=3)),
        ('sdc_parallel', dict(M=3, **PARALLEL)),
        ('sdc_parallel', dict(M=4, **PARALLEL)),
    ]:
        cs += [(label, 'sdc', dt, dict(opts, restol=tol)) for dt in [0.5, 0.25, 0.125] for tol in [1e-6, 1e-8, 1e-10]]
    return cs


def tag(c):
    return '_'.join([c[0], f'{c[2]:g}'] + [f'{k}{v}' for k, v in sorted(c[3].items())])


def run_one(args):
    c, setup = args
    label, method, dt, opts = c
    if method == 'sdc':
        res = run_sdc(dt, setup['Tend'], setup['dt_out'], problem_params=setup['problem'], **opts)
    else:
        res = run_propagator(method, dt, setup['Tend'], setup['dt_out'], problem_params=setup['problem'], **opts)
    return dict(
        tag=tag(c),
        label=label,
        method=method,
        dt=dt,
        opts=opts,
        **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items()},
    )


def run_all(full, outdir, procs=1):
    """Run every configuration not yet in outdir/runs.json (one JSON object per line) and return all results"""
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, 'runs.json')
    done = {}
    if os.path.exists(fname):
        for line in open(fname):
            r = json.loads(line)
            done[r['tag']] = r
    setup = FULL if full else DEMO
    todo = [c for c in configs(full) if tag(c) not in done]
    with Pool(procs) if procs > 1 else _Serial() as pool:
        for r in pool.imap_unordered(run_one, [(c, setup) for c in todo]):
            done[r['tag']] = r
            with open(fname, 'a') as fh:
                fh.write(json.dumps(r) + '\n')
    return [done[tag(c)] for c in configs(full)]


class _Serial:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def imap_unordered(self, f, it):
        return map(f, it)


def harmonic_yields(t, d, omega=0.057, qmax=25):
    """Odd harmonic yields of the dipole acceleration spectrum |w^2 d(w)|^2"""
    t, d = np.asarray(t), np.asarray(d)
    w = 2 * np.pi * np.fft.rfftfreq(len(t), t[1] - t[0])
    S = np.abs(np.fft.rfft((d - d.mean()) * np.sin(np.pi * t / t[-1]) ** 2)) ** 2 * w**4
    return np.array([S[(w >= (q - 1) * omega) & (w < (q + 1) * omega)].sum() for q in range(1, qmax + 1, 2)])


def add_errors(rows):
    ref = rows[0]
    for r in rows[1:]:
        d, dr = np.asarray(r['d']), np.asarray(ref['d'])
        if len(d) != len(dr) or not np.all(np.isfinite(d)):
            r['err'] = r['err_hhg'] = np.inf
            continue
        r['err'] = float(np.max(np.abs(d - dr)) / np.max(np.abs(dr)))
        r['err_hhg'] = float(np.max(np.abs(np.log10(harmonic_yields(r['t'], d) / harmonic_yields(ref['t'], dr)))))
    return rows[1:]


def front(rows, label):
    """Cheapest runs of one family that improve the error"""
    out, best = [], np.inf
    for r in sorted([r for r in rows if r['label'] == label], key=lambda r: r['cost']):
        if r['err'] < best:
            out.append(r)
            best = r['err']
    return out


SERIES = [  # label, legend, color, marker, linestyle
    ('sdc_parallel', 'SDC, nodes in parallel (critical path)', '#2a78d6', 'o', '-'),
    ('cfm4', 'commutator-free Magnus 4, Lanczos', '#eb6834', 'D', '-'),
    ('sdc_serial', 'SDC, serial', '#1baf7a', 's', '-'),
    ('aetrs', 'AETRS, Taylor 4 (Octopus default)', '#eda100', '^', '-'),
    ('crank_nicolson', 'Crank-Nicolson, GMRES', '#e87ba4', 'v', '-'),
    ('strang', 'split-operator 2 (exact projector exponentials)', '#6b6a63', 'x', '--'),
]


def plot(rows, reference, fname):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3df'
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw=dict(width_ratios=[1, 1.35], wspace=0.28))
    t, d = np.asarray(reference['t']), np.asarray(reference['d'])
    w = 2 * np.pi * np.fft.rfftfreq(len(t), t[1] - t[0])
    S = np.abs(np.fft.rfft((d - d.mean()) * np.sin(np.pi * t / t[-1]) ** 2)) ** 2 * w**4
    sel = (w > 0.3 * 0.057) & (w < 45 * 0.057)
    ax0.semilogy(w[sel] / 0.057, S[sel], color='#2a78d6', lw=1.2)
    ax0.set_xlabel('harmonic order', color=INK)
    ax0.set_ylabel('harmonic spectrum |ω² d(ω)|²', color=INK)
    ax0.set_title('1D diatomic, 3 orbitals, nonlocal pseudopotential', color=INK, fontsize=10, loc='left')
    for label, legend, col, mk, ls in SERIES:
        fr = np.array([(r['cost'], r['err']) for r in front(rows, label) if r['err'] < 3e-2])
        if len(fr):
            ax1.loglog(
                fr[:, 0],
                fr[:, 1],
                ls=ls,
                color=col,
                marker=mk,
                ms=6,
                lw=2 if ls == '-' else 1.4,
                mec='white' if mk != 'x' else col,
                mew=1.0,
                label=legend,
            )
    ax1.set_xlabel('cost [orbital FFT round trips = H applications]', color=INK)
    ax1.set_ylabel('max dipole error (relative)', color=INK)
    ax1.set_title('work-precision over the laser pulse', color=INK, fontsize=10, loc='left')
    ax1.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2)
    for ax in (ax0, ax1):
        ax.grid(True, which='major', color=GRID, lw=0.8)
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
        for s in ['left', 'bottom']:
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED)
    fig.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close(fig)


def main(full=False, procs=1):
    outdir = 'data/tddft_full' if full else 'data/tddft_demo'
    allrows = run_all(full, outdir, procs)
    rows = add_errors(allrows)
    plot(rows, allrows[0], f'data/tddft_work_precision{"_full" if full else ""}.png')
    for label, *_ in SERIES:
        for r in front(rows, label):
            print(
                f'{label:16s} dt={r["dt"]:<7g} cost {r["cost"]:8d}  dipole error {r["err"]:.1e}  harmonics {r["err_hhg"]:.3f} dec'
            )
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--full', action='store_true', help='the full set behind the README numbers (hours)')
    parser.add_argument('--procs', type=int, default=1, help='runs at once')
    a = parser.parse_args()
    main(a.full, a.procs)
