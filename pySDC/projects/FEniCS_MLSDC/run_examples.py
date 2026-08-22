"""
Run the three FEniCS examples with SDC, MLSDC (2 and 3 levels) and PFASST.

Each example is run with continuous (``CG``) and discontinuous (``DG``) elements, and each of those
with h-coarsening (coarser mesh, same element order) and p-coarsening (same mesh, lower element
order). Per example and family this produces:

* iterations for SDC against MLSDC with 2 and 3 levels, once per coarsening direction, plus the work
  each costs, to show which hierarchy actually pays;
* iterations for PFASST over a growing number of parallel steps, to show they stay bounded.

Run with ``python run_examples.py``; results are written to ``data/fenics_mlsdc_out.txt``.
"""

from pathlib import Path

import numpy as np

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.FEniCS_MLSDC.setups import COARSENINGS, EXAMPLES, FAMILIES, get_description, get_pfasst_procs


def run(example, nlevels=1, num_procs=1, **kwargs):
    """
    Run one configuration.

    Returns
    -------
    dict
        ``niter`` (mean over the steps), ``uend``, ``dofs`` per level and ``work``, the number of
        fine-level sweep equivalents: iterations times the summed dof ratio of the hierarchy.
    """
    description, controller_params, t0, Tend = get_description(example, nlevels=nlevels, **kwargs)
    controller = controller_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)

    step = controller.MS[0]
    dofs = [level.prob.init.dim() for level in step.levels]
    prob = step.levels[0].prob

    uend, stats = controller.run(u0=prob.u_exact(t0), t0=t0, Tend=Tend)
    niter = np.mean([item[1] for item in get_sorted(stats, type='niter', sortby='time')])

    return {
        'niter': niter,
        'uend': uend,
        'dofs': dofs,
        'work': niter * sum(n / dofs[0] for n in dofs),
    }


def compare_mlsdc(example, family='CG', coarsening='h', out=print, **kwargs):
    """SDC against MLSDC with 2 and 3 levels. Returns the per-level results keyed by nlevels."""
    results = {
        nlevels: run(example, nlevels=nlevels, family=family, coarsening=coarsening, **kwargs) for nlevels in (1, 2, 3)
    }
    ref = results[1]

    ladder = 'mesh coarsening' if coarsening == 'h' else 'order coarsening'
    out(f'\n{example} [{family}, {coarsening}]: SDC vs MLSDC ({ladder}, collocation nodes kept)')
    out(f'  {"levels":>7s} {"dofs":>22s} {"niter":>7s} {"work":>7s} {"speed-up":>9s} {"|u - u_SDC|":>12s}')
    for nlevels, res in results.items():
        speedup = ref['work'] / res['work']
        diff = abs(res['uend'] - ref['uend'])
        out(
            f'  {nlevels:7d} {str(res["dofs"]):>22s} {res["niter"]:7.2f} {res["work"]:7.2f} '
            f'{speedup:8.2f}x {diff:12.2e}'
        )
    return results


def check_pfasst(example, nlevels=2, family='CG', coarsening='h', procs=None, out=print, **kwargs):
    """PFASST over a growing number of parallel steps. Returns the results keyed by num_procs."""
    procs = get_pfasst_procs(example) if procs is None else procs
    results = {
        p: run(example, nlevels=nlevels, num_procs=p, family=family, coarsening=coarsening, **kwargs) for p in procs
    }
    ref = results[procs[0]]

    out(f'\n{example} [{family}, {coarsening}]: PFASST with {nlevels} levels, iterations over parallel steps')
    out(f'  {"procs":>6s} {"niter":>7s} {"|u - u_serial|":>15s}')
    for p, res in results.items():
        out(f'  {p:6d} {res["niter"]:7.2f} {abs(res["uend"] - ref["uend"]):15.2e}')
    return results


def main():
    Path('data').mkdir(parents=True, exist_ok=True)
    with open('data/fenics_mlsdc_out.txt', 'w') as f:

        def out(line=''):
            print(line)
            f.write(str(line) + '\n')

        out('FEniCS + pySDC, mass-matrix formulation throughout (no mass inverse anywhere).')
        out('work = iterations x sum(dofs_l / dofs_0), i.e. fine-level sweep equivalents.')
        out('[family, coarsening]: CG/DG elements, h = coarser mesh, p = lower element order.')
        for example in EXAMPLES:
            for family in FAMILIES:
                for coarsening in COARSENINGS:
                    compare_mlsdc(example, family=family, coarsening=coarsening, out=out)
                    check_pfasst(example, family=family, coarsening=coarsening, out=out)


if __name__ == '__main__':
    main()
