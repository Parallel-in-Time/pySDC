"""
The numerical experiments of Binkowski, Christian, Götschel, Minion: "Toward Spectral Deferred Correction Methods for
Stochastic Differential Equations" (arXiv:2610.23693), with SDC-SBB in pySDC.

Run all of them with the sample sizes of the paper (this takes a while) by

.. code-block:: bash

    python pySDC/projects/SDC4SDEs/studies.py

which writes the plots to ``data/`` and prints the fitted orders. Each study is a function with smaller defaults
available through its arguments, which the tests use.

"SDC-M" is SDC-SBB on M Lobatto nodes with 2(M - 1) sweeps, whose expected strong and weak order is M - 1: every
second sweep gains one order, up to half the order 2M - 2 of the collocation method.
"""

import os

import numpy as np

from pySDC.projects.SDC4SDEs.brownian import brownian_paths, smooth_paths
from pySDC.projects.SDC4SDEs.problem_classes import (
    LinearScalarSDE,
    LinearSystemSDE,
    NonlinearScalarSDE,
    OrnsteinUhlenbeck,
)
from pySDC.projects.SDC4SDEs.run import convergence, order, solve_sdc

# problem class and initial guess of SDC-SBB per example, as in the paper
PROBLEMS = {
    'linear': (LinearScalarSDE, 'euler_maruyama'),
    'nonlinear': (NonlinearScalarSDE, 'euler'),
    'system': (LinearSystemSDE, 'euler'),
}
# categorical colours in fixed order, and an ordinal blue ramp from light to dark
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
BLUES = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']


def _figure(nrows, ncols):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.grid(color='#e6e6e6', linewidth=0.6)
        ax.spines[['top', 'right']].set_visible(False)
    return fig, axes


def _slopes(ax, x, anchor, orders):
    """Dashed grey reference lines through the point anchor at the largest x, labelled with their order."""
    x = np.asarray(x)
    for p in orders:
        y = anchor * (x / x.max()) ** p
        ax.plot(x, y, '--', color='#9a9a94', linewidth=0.9)
        ax.annotate(f'{p:g}', (x.min(), y[np.argmin(x)]), color='#6e6e68', fontsize=8, ha='right', va='center')


def _save(fig, name):
    os.makedirs('data', exist_ok=True)
    path = os.path.join('data', name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path


def sweeps_and_order(n_samples=10**4, num_nodes=5, max_sweeps=8, dts=None, seed=0, plot=True):
    """
    Strong error of SDC-SBB on the linear scalar SDE for 0 to max_sweeps sweeps, for the two initial guesses of the
    paper (its Figure 1): every second sweep gains one order.

    Returns:
        dict: per initial guess, the strong errors of shape (max_sweeps + 1, len(dts)), and the step sizes 'dt'
    """
    dts = [2.0**-k for k in range(3, 7)] if dts is None else dts
    params = {'brownian_path': brownian_paths(n_samples, round(1 / min(dts)), seed=seed)}
    result = {'dt': np.asarray(dts)}
    for initial_guess in ['euler', 'euler_maruyama']:
        method = {'num_nodes': num_nodes, 'initial_guess': initial_guess}
        result[initial_guess] = np.array(
            [
                convergence({**method, 'num_sweeps': k}, LinearScalarSDE, params, dts)['strong'][:, 0]
                for k in range(max_sweeps + 1)
            ]
        )
    if plot:
        fig, axes = _figure(1, 2)
        titles = {'euler': 'Euler initial guess', 'euler_maruyama': 'Euler-Maruyama initial guess'}
        for ax, initial_guess in zip(axes[0], ['euler', 'euler_maruyama'], strict=True):
            for k, err in enumerate(result[initial_guess]):
                label = 'initial guess' if k == 0 else f'{k} sweep{"s" * (k > 1)}'
                ax.plot(dts, err, 'o-', color=BLUES[k], linewidth=1.5, markersize=4, label=label)
            _slopes(ax, dts, result[initial_guess][-1, 0] * 3, [4])
            _slopes(ax, dts, result[initial_guess][2, 0] * 3, [1])
            ax.set(title=f'{titles[initial_guess]}, M = {num_nodes}', xlabel=r'$\Delta T$', ylabel='strong error')
        axes[0, 1].legend(fontsize=7, frameon=False, loc='upper left', bbox_to_anchor=(1.0, 1.0))
        result['plot'] = _save(fig, 'SDC4SDEs_sweeps_and_order.png')
    return result


def convergence_and_work(problem, n_samples=10**4, dts=None, nodes=(3, 4, 5), seed=0, plot=True):
    """
    Strong and weak errors of Euler-Maruyama, Milstein and SDC-M against the step size and the work (evaluations of
    drift, diffusion and its derivative per sample), for one of the examples of the paper (its Figures 2 to 6).

    Args:
        problem (str): 'linear', 'nonlinear' or 'system'
        n_samples (int): number of Brownian paths, the same for all methods
        dts (list): step sizes, by default 2^-1 to 2^-10 (2^-2 to 2^-10 for the nonlinear SDE, where explicit
            methods blow up on some paths with 2^-1)
        nodes (tuple): numbers M of Lobatto nodes of SDC-M
        seed: seed of the Brownian paths

    Returns:
        dict: per method, the output of :func:`pySDC.projects.SDC4SDEs.run.convergence`
    """
    problem_class, initial_guess = PROBLEMS[problem]
    if dts is None:
        dts = [2.0**-k for k in range(2 if problem == 'nonlinear' else 1, 11)]
    params = {'brownian_path': brownian_paths(n_samples, round(1 / min(dts)), n_noise=problem_class.n_noise, seed=seed)}
    methods = {'Euler-Maruyama': 'EM', 'Milstein': 'Milstein'}
    for M in nodes:
        methods[f'SDC-{M}'] = {'num_nodes': M, 'num_sweeps': 2 * (M - 1), 'initial_guess': initial_guess}
    result = {name: convergence(method, problem_class, params, dts) for name, method in methods.items()}
    if plot:
        ndim = problem_class.ndim
        fig, axes = _figure(ndim, 4)
        for i in range(ndim):
            component = f', component {i + 1}' if ndim > 1 else ''
            for ax, kind, x in zip(
                axes[i], ['strong', 'weak', 'strong', 'weak'], ['dt', 'dt', 'work', 'work'], strict=True
            ):
                for (name, r), color in zip(result.items(), COLORS, strict=False):
                    ok = np.isfinite(r[kind][:, i]) & (r[kind][:, i] > 0)
                    ax.plot(r[x][ok], r[kind][ok, i], 'o-', color=color, linewidth=1.5, markersize=4, label=name)
                if x == 'work':
                    ax.set_xscale('log', base=10)
                    ax.set(xlabel='evaluations per sample')
                else:
                    ax.set(xlabel=r'$\Delta T$')
                ax.set(ylabel=f'{kind} error', title=f'{kind} error{component}')
        axes[0, -1].legend(fontsize=7, frameon=False, loc='upper left', bbox_to_anchor=(1.0, 1.0))
        result['plot'] = _save(fig, f'SDC4SDEs_convergence_{problem}.png')
    return result


def ornstein_uhlenbeck(
    n_samples=1000, n_fine=2**14, n_modes=64, n_bridge_terms=128, num_nodes=4, dts=None, seed=0, plot=True
):
    """
    Beyond the linear bridge (the paper's Figure 7): the Ornstein-Uhlenbeck process needs the sine terms of the
    bridge.

    The paths are first replaced by their Karhunen-Loève expansion with n_modes terms on [0, 1], which is smooth.
    SDC-SBB with n_bridge_terms terms is then compared with the exact solution for the smooth path and with the one
    for the rough path, as is the exact solution of the random ODE (the error of the bridge alone). In addition,
    SDC-SBB is run on the rough path itself, with and without the sine terms.

    Returns:
        dict: arrays of strong errors per curve, and 'dt'
    """
    dts = [2.0**-k for k in range(2, 8)] if dts is None else dts
    rough = brownian_paths(n_samples, n_fine, seed=seed)
    smooth = smooth_paths(rough, 1.0, n_modes)
    reference = {path: OrnsteinUhlenbeck(W).u_exact(1.0) for path, W in [('smooth', smooth), ('rough', rough)]}
    sdc = {'num_nodes': num_nodes, 'num_sweeps': 2 * (num_nodes - 1), 'initial_guess': 'spread'}
    J = n_bridge_terms

    def error(u, path):
        return np.mean(np.abs(np.asarray(u) - reference[path]))

    curves = {
        f'random ODE, J = {J}, vs. smooth path': [],
        f'random ODE, J = {J}, vs. rough path': [],
        f'SDC-{num_nodes}, J = {J}, vs. smooth path': [],
        f'SDC-{num_nodes}, J = {J}, vs. rough path': [],
        f'SDC-{num_nodes} on rough path, J = 0': [],
        f'SDC-{num_nodes} on rough path, J = {J}': [],
    }
    lists = list(curves.values())
    for dt in dts:
        u = OrnsteinUhlenbeck(smooth, n_bridge_terms=J).solve_sbb_exactly(dt, 1.0)
        lists[0].append(error(u, 'smooth'))
        lists[1].append(error(u, 'rough'))
        u = solve_sdc(OrnsteinUhlenbeck, {'brownian_path': smooth, 'n_bridge_terms': J}, dt, **sdc)[0]
        lists[2].append(error(u, 'smooth'))
        lists[3].append(error(u, 'rough'))
        for k, terms in [(4, 0), (5, J)]:
            u = solve_sdc(OrnsteinUhlenbeck, {'brownian_path': rough, 'n_bridge_terms': terms}, dt, **sdc)[0]
            lists[k].append(error(u, 'rough'))
    result = {'dt': np.asarray(dts), **{name: np.asarray(e) for name, e in curves.items()}}
    if plot:
        fig, axes = _figure(1, 1)
        ax = axes[0, 0]
        for (name, e), color, style in zip(curves.items(), COLORS, ['o-', 'o-', 's-', 's-', 'd:', 'd:'], strict=False):
            ax.plot(dts, e, style, color=color, linewidth=1.5, markersize=4, label=name)
        _slopes(ax, dts, result[f'SDC-{num_nodes}, J = {J}, vs. smooth path'][0] * 3, [1, 3])
        ax.set(xlabel=r'$\Delta T$', ylabel='strong error', title='Ornstein-Uhlenbeck process')
        ax.legend(fontsize=7, frameon=False, loc='lower right')
        fig.set_size_inches(6.5, 4.5)
        result['plot'] = _save(fig, 'SDC4SDEs_ornstein_uhlenbeck.png')
    return result


def print_orders(name, result, kinds=('strong', 'weak')):
    """Print the fitted orders of the methods of a study from convergence_and_work."""
    print(name)
    for method, r in result.items():
        if method == 'plot':
            continue
        fits = []
        for kind in kinds:
            e = r[kind][:, 0]
            # fit only where the error is above round-off and finite
            ok = np.isfinite(e) & (e > 1e-12)
            fits.append(f'{kind} {order(r["dt"][ok], e[ok]):5.2f}' if ok.sum() > 1 else f'{kind}   -  ')
        print(f'  {method:15s} ' + '  '.join(fits))


if __name__ == '__main__':
    result = sweeps_and_order()
    print('sweeps and order, strong orders per number of sweeps:')
    for initial_guess in ['euler', 'euler_maruyama']:
        print(f'  {initial_guess:15s}', ' '.join(f'{order(result["dt"], e):5.2f}' for e in result[initial_guess]))
    for problem in PROBLEMS:
        print_orders(f'{problem} SDE, orders of component 1:', convergence_and_work(problem))
    result = ornstein_uhlenbeck()
    print('Ornstein-Uhlenbeck, strong orders:')
    for name, e in result.items():
        if name not in ['dt', 'plot']:
            print(f'  {name:42s} {order(result["dt"], e):5.2f}')
