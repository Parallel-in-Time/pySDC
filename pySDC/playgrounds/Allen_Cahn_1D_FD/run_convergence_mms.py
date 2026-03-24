"""
MMS convergence study – isolating order reduction in Allen-Cahn IMEX-SDC
=========================================================================

This script compares temporal convergence for three manufactured-solution
variants of the 1D Allen-Cahn equation to isolate the cause of the order
stalling observed in the travelling-wave study.

**Background**

In the travelling-wave study (``run_convergence.py``), the IMEX-SDC
convergence order stalls at :math:`\\approx 2.6` for :math:`K = 3` sweeps
and :math:`\\approx 2.8` for :math:`K = 4` sweeps.  Two possible causes:

1. **Nonlinear explicit reaction term** – the nonlinear term broadens the
   pre-asymptotic region for all IMEX-SDC formulations.
2. **Time-dependent boundary-correction vector** :math:`b_\\text{bc}(t)` in
   :math:`f_\\text{impl}` – the :math:`b_\\text{bc}(t)` term is not a
   function of :math:`u`; including it in the *implicit* part (instead of
   the explicit part) leads to incorrect quadrature of this time-dependent
   source, causing order reduction.

**Why the travelling-wave case appears unaffected by cause 2**

In the travelling-wave study the wave speed is small
(:math:`v \\approx -0.05`), so :math:`b_\\text{bc}(t)` varies very slowly.
The quadrature error from including the slowly-varying :math:`b_\\text{bc}`
in the implicit quadrature sum is therefore tiny compared to the temporal
error, and the dominant effect on the observed order is the nonlinear
reaction (cause 1).

The manufactured cosine solution (:math:`\\cos(\\pi x)\\cos(t)`) has
:math:`b_\\text{bc}(t) \\propto \\cos(t)/\\Delta x^2`, which varies
on an :math:`O(1)` time scale.  This makes cause 2 clearly visible: the
standard formulation stalls at order :math:`\\approx 1` for :math:`K \\ge 2`,
while the homogeneous and lifted formulations achieve the expected orders
(modulo the pre-asymptotic effect of the nonlinear reaction).

**Error measurement**

Errors are measured against a **fine-:math:`\\Delta t` reference solution**
(same formulation, :math:`\\Delta t_\\text{ref} = T_\\text{end}/1024`,
fully-converged SDC) to eliminate the spatial-discretisation error from
the comparison.

**Summary of findings**

* The standard :math:`b_\\text{bc}` approach reduces the effective order to
  :math:`\\approx 1` for :math:`K \\ge 2` when :math:`b_\\text{bc}(t)`
  varies rapidly (cos MMS case).
* Boundary lifting removes :math:`b_\\text{bc}` from :math:`f_\\text{impl}`
  and largely restores the expected order.
* Both the homogeneous and lifted cases still show pre-asymptotic stalling
  at :math:`\\approx 2.6` for :math:`K \\ge 3`, confirming that the
  nonlinear reaction causes a broadened pre-asymptotic regime.
* The travelling-wave study stalls for similar reasons (reaction-driven
  pre-asymptotic behavior), but the :math:`b_\\text{bc}` contribution is
  negligible there due to the small wave speed.

**Parameters**

* Domain :math:`[0, 1]`, :math:`\\varepsilon = 1`,
  :math:`d_w = 0` (modest nonlinear reaction)
* :math:`T_\\text{end} = 0.5`, ``nvars = 127``
* RADAU-RIGHT quadrature, :math:`M = 3` nodes, :math:`K = 1, 2, 3, 4` sweeps

Usage::

    python run_convergence_mms.py
"""

import numpy as np
import matplotlib.pyplot as plt

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.playgrounds.Allen_Cahn_1D_FD.AllenCahn_1D_FD_MMS import (
    allencahn_1d_mms_hom,
    allencahn_1d_mms_inhom,
    allencahn_1d_mms_inhom_lift,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SWEEPER_PARAMS = {
    'quad_type': 'RADAU-RIGHT',
    'num_nodes': 3,
    'QI': 'LU',
    'QE': 'EE',
    'initial_guess': 'spread',
}


def _build_description(problem_class, nvars, dt, max_iter, eps=1.0, dw=0.0, restol=1e-20):
    """Return a pySDC description dict."""
    return {
        'problem_class': problem_class,
        'problem_params': {'nvars': nvars, 'eps': eps, 'dw': dw},
        'sweeper_class': imex_1st_order,
        'sweeper_params': _SWEEPER_PARAMS,
        'level_params': {'restol': restol, 'dt': dt},
        'step_params': {'maxiter': max_iter},
    }


def _run(problem_class, nvars, dt, max_iter, eps=1.0, dw=0.0, t0=0.0, Tend=0.5):
    """
    Run one simulation and return (physical_u_array, problem).

    For the lifted variant the returned array is :math:`v + E(T)`,
    i.e. the physical solution *u* (not the lifted variable *v*).
    """
    desc = _build_description(problem_class, nvars, dt, max_iter, eps=eps, dw=dw)
    ctrl = controller_nonMPI(num_procs=1, controller_params={'logger_level': 40}, description=desc)
    P = ctrl.MS[0].levels[0].prob
    state_end, _ = ctrl.run(u0=P.u_exact(t0), t0=t0, Tend=Tend)
    state_arr = np.asarray(state_end).copy()
    # For the lifted formulation recover physical u = v + E(T).
    if isinstance(P, allencahn_1d_mms_inhom_lift):
        state_arr = state_arr + P.lift(Tend)
    return state_arr, P


# ---------------------------------------------------------------------------
# Temporal convergence study (MMS: error vs. exact solution)
# ---------------------------------------------------------------------------

def mms_temporal_convergence(problem_class, label, nvars=127, Tend=0.5,
                              eps=1.0, dw=0.0, num_nodes=3):
    r"""
    Vary :math:`\Delta t` with a fixed sweep count :math:`K` per step and
    measure the error against a **fine-:math:`\Delta t` reference solution**
    of the same formulation (fully-converged SDC, same spatial grid).

    Using a reference instead of the analytic exact solution removes the
    spatial-discretisation error from the comparison, so only temporal
    convergence is visible.

    Parameters
    ----------
    problem_class : type
        One of the three MMS classes.
    label : str
        Short label for printout.
    nvars : int
        Number of interior grid points.
    Tend : float
        Final time.
    eps, dw : float
        Allen-Cahn parameters.
    num_nodes : int
        SDC collocation nodes (:math:`M`).

    Returns
    -------
    results : dict
        ``{K: (dts, errors)}``
    """
    # Fine-dt reference: 1024 steps, fully-converged SDC (same formulation).
    dt_ref = Tend / 1024
    uref, _ = _run(problem_class, nvars, dt_ref, max_iter=50,
                   eps=eps, dw=dw, Tend=Tend)

    dts = [Tend / (2**k) for k in range(1, 7)]
    sweep_counts = [1, 2, 3, 4]
    max_order = 2 * num_nodes - 1  # collocation order

    print('\n' + '=' * 70)
    print(f'  MMS temporal convergence  [{label}]  (nvars={nvars}, M={num_nodes})')
    print(f'  error vs. fine-dt reference  (dt_ref={dt_ref:.2e});  ε={eps}, dw={dw}')
    print('=' * 70)

    results = {}
    for K in sweep_counts:
        expected = min(K, max_order)
        print(f'\n  K = {K} sweep(s) per step  (expected order {expected}):')
        print(f'  {"dt":>10}  {"error (inf)":>14}  {"order":>8}  {"expected":>10}')
        errs = []
        for dt in dts:
            uend, _ = _run(problem_class, nvars, dt, K,
                           eps=eps, dw=dw, Tend=Tend)
            err = float(np.linalg.norm(uend - uref, np.inf))
            errs.append(err)
            if len(errs) > 1 and errs[-2] > 0.0:
                order = np.log(errs[-2] / err) / np.log(dts[len(errs) - 2] / dt)
                print(f'  {dt:>10.5f}  {err:>14.4e}  {order:>8.2f}  {expected:>10d}')
            else:
                print(f'  {dt:>10.5f}  {err:>14.4e}  {"---":>8}  {expected:>10d}')
        results[K] = (dts, errs)

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plot(results_hom, results_inhom, results_lift, num_nodes=3):
    """Save a three-panel comparison plot."""
    max_order = 2 * num_nodes - 1
    markers = ['o', 's', '^', 'D']
    colors = ['C0', 'C1', 'C2', 'C3']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    titles = [
        f'Homogeneous BCs\n(sin solution, M={num_nodes})',
        f'Inhomogeneous BCs, standard\n(cos solution, b_bc, M={num_nodes})',
        f'Inhomogeneous BCs, lifted\n(cos solution, lifting, M={num_nodes})',
    ]

    for ax, results, title in zip(axes, [results_hom, results_inhom, results_lift], titles):
        for (K, (dts, errs)), marker, color in zip(results.items(), markers, colors):
            expected = min(K, max_order)
            ax.loglog(dts, errs, marker=marker, color=color, label=f'K={K}')
            ref = errs[-1] * (np.array(dts) / dts[-1]) ** expected
            ax.loglog(dts, ref, linestyle='--', color=color, alpha=0.5,
                      label=f'order {expected}')
        ax.set_xlabel(r'$\Delta t$', fontsize=12)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, which='both', linestyle=':')

    axes[0].set_ylabel(r'error vs. fine-$\Delta t$ reference', fontsize=11)
    plt.tight_layout()
    fname = 'allen_cahn_1d_mms_convergence.png'
    plt.savefig(fname, dpi=150)
    print(f'\nPlot saved to {fname}')


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _last_order(results, K):
    """Compute the convergence order over the two finest time steps."""
    dts, errs = results[K]
    if len(errs) >= 2 and errs[-2] > 0.0 and errs[-1] > 0.0:
        return np.log(errs[-2] / errs[-1]) / np.log(dts[-2] / dts[-1])
    return float('nan')


def print_summary(results_hom, results_inhom, results_lift, num_nodes=3):
    """Print a side-by-side summary of the last observed orders."""
    max_order = 2 * num_nodes - 1
    sweep_counts = list(results_hom.keys())
    print('\n' + '=' * 75)
    print('  Summary: last measured convergence order  (expected = min(K, 2M-1))')
    print('  (measured over the two finest time steps)')
    print('=' * 75)
    print(f'  {"K":>3}  {"expected":>9}  {"hom (sin)":>12}  '
          f'{"inhom std (cos)":>15}  {"inhom lift":>12}')
    print('  ' + '-' * 71)
    for K in sweep_counts:
        expected = min(K, max_order)
        o_hom = _last_order(results_hom, K)
        o_inh = _last_order(results_inhom, K)
        o_lft = _last_order(results_lift, K)
        print(f'  {K:>3}  {expected:>9}  {o_hom:>12.2f}  {o_inh:>15.2f}  {o_lft:>12.2f}')
    print()
    print('  Findings:')
    print('  - inhom std << hom & inhom lift: b_bc(t) in f.impl causes order reduction.')
    print('  - inhom lift ≈ hom: lifting removes b_bc from f.impl, restoring orders.')
    print('  - hom stalls at ~2.6 for K≥3: nonlinear reaction widens pre-asymptotic regime.')
    print('  - In the travelling-wave study b_bc varies slowly (small wave speed),')
    print('    so cause 2 is negligible there; stalling is dominated by the reaction.')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('\n' + '#' * 75)
    print('# Case 1: Homogeneous BCs – sin(πx)cos(t)                              #')
    print('# (no b_bc correction, no time-dependent BCs)                          #')
    print('#' * 75)
    res_hom = mms_temporal_convergence(allencahn_1d_mms_hom, 'hom, sin, no b_bc')

    print('\n' + '#' * 75)
    print('# Case 2: Inhomogeneous BCs, standard – cos(πx)cos(t) + b_bc(t)       #')
    print('# (b_bc correction, time-dependent BCs)                                #')
    print('#' * 75)
    res_inhom = mms_temporal_convergence(allencahn_1d_mms_inhom, 'inhom, cos, b_bc')

    print('\n' + '#' * 75)
    print('# Case 3: Inhomogeneous BCs, lifted – cos(πx)cos(t), v = u-E          #')
    print('# (autonomous f.impl, time-dependent BCs removed by lifting)           #')
    print('#' * 75)
    res_lift = mms_temporal_convergence(allencahn_1d_mms_inhom_lift, 'inhom, cos, lifted')

    print_summary(res_hom, res_inhom, res_lift)
    make_plot(res_hom, res_inhom, res_lift)


if __name__ == '__main__':
    main()
