import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.playgrounds.time_dep_BCs.heat_time_dep_BCs import Heat1DTimeDepBCs, generic_implicit_diffbc

# label: (ft, bc_mode); constant data is the reference for the design order 2M-1
VARIANTS = {
    'constant': (0, 'pointwise'),
    'pointwise': (1, 'pointwise'),
    'lifted': (1, 'lifted'),
    'differentiated': (1, 'differentiated'),
}


def run(dt, num_nodes, ft, bc_mode, Tend=4, maxiter=30):
    """Solve up to `Tend` with converged SDC and return the solution u (not v) at `Tend`"""
    description = {
        'problem_class': Heat1DTimeDepBCs,
        'problem_params': {'ft': ft, 'bc_mode': bc_mode, 'spectral_space': False},
        'sweeper_class': generic_implicit_diffbc if bc_mode == 'differentiated' else generic_implicit,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': num_nodes, 'QI': 'LU'},
        'level_params': {'dt': dt, 'restol': 1e-13},
        'step_params': {'maxiter': maxiter},
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    P = controller.MS[0].levels[0].prob
    uend, _ = controller.run(u0=P.u_exact(0), t0=0, Tend=Tend)
    return P.to_u(uend, Tend)


def convergence(num_nodes, ft, bc_mode, n_dt=7, Tend=4):
    """Step sizes, and max-norm differences on the grid between consecutive step sizes"""
    dts = [Tend / 2 ** (n + 1) for n in range(n_dt + 1)]
    u = [run(dt, num_nodes, ft, bc_mode, Tend) for dt in dts]
    errors = [abs(u[i] - u[i + 1]) / abs(u[-1]) for i in range(n_dt)]
    return dts[:-1], errors, u[-1]


def orders(errors, floor=1e-12):
    """Observed orders between consecutive step sizes, stopping at the roundoff floor"""
    return [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1) if errors[i + 1] > floor]


def main(num_nodes=(3, 4)):  # pragma: no cover
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, len(num_nodes), sharey=True, figsize=(4.5 * len(num_nodes), 4))
    for ax, M in zip(np.atleast_1d(axs), num_nodes):
        finest = {}
        for label, (ft, bc_mode) in VARIANTS.items():
            dts, errors, finest[label] = convergence(M, ft, bc_mode)
            p = orders(errors)
            print(f'M={M} {label:15s} design={2 * M - 1} orders: ' + ' '.join(f'{me:.2f}' for me in p))
            ax.loglog(dts, errors, marker='.', label=f'{label}, order {np.median(p[-3:]):.1f}')
        for label in ['lifted', 'differentiated']:
            print(f'M={M} |{label} - pointwise| at finest dt: {abs(finest[label] - finest["pointwise"]):.1e}')
        ax.set_title(f'RADAU-RIGHT, M={M}, design order {2 * M - 1}')
        ax.set_xlabel(r'$\Delta t$')
        ax.legend(frameon=False)
    np.atleast_1d(axs)[0].set_ylabel('relative difference to dt/2')
    fig.tight_layout()
    fig.savefig('time_dep_BCs_order.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
