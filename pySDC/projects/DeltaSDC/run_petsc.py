r"""
Delta-form SDC on the PETSc Generalized Fisher problem.

Runnable entry point in the style of ``pySDC/tutorial/step_7``: the logic lives here and the
PETSc-marked test simply calls :func:`main`.

Reduced precision is **emulated** -- PETSc fixes its scalar type at build time, so values are
rounded through the working precision while the arithmetic stays at the backend type.
"""

import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.GeneralizedFisher_1D_PETSc import petsc_fisher_fullyimplicit
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.transfer_classes.TransferPETScDMDA import mesh_to_mesh_petsc_dmda
from pySDC.projects.DeltaSDC.cascade import delta_implicit_cascade
from pySDC.projects.DeltaSDC.mlsdc import (
    delta_implicit_rounded,
    delta_transfer,
    rounding_transfer,
)
from pySDC.projects.DeltaSDC.problems_petsc import petsc_fisher_delta
from pySDC.implementations.sweeper_classes.delta_form import delta_implicit

COARSE_NVARS = 64
"""Fine level is 2 * 64 - 1 = 127 points, which is what a DMDA injection needs."""

PROBLEM_PARAMS = {
    'nvars': 127,
    'lambda0': 2.0,
    'nu': 1,
    'interval': (-5.0, 5.0),
    'lsol_tol': 1e-12,
    'nlsol_tol': 1e-12,
    'lsol_maxiter': 200,
    'nlsol_maxiter': 50,
}

SWEEPER_PARAMS = {
    'quad_type': 'RADAU-RIGHT',
    'node_type': 'LEGENDRE',
    'num_nodes': 3,
    'QI': 'LU',
    'initial_guess': 'spread',
}


def run(
    problem_class,
    problem_params,
    sweeper_class,
    sweeper_extra=None,
    multilevel=False,
    base_transfer_class=None,
    maxiter=10,
    dt=0.1,
    nsteps=2,
):
    """
    Run a short Fisher simulation.

    Parameters
    ----------
    problem_class : type
        Problem class to integrate.
    problem_params : dict
        Parameters for the problem class.
    sweeper_class : type
        Sweeper class to use.
    sweeper_extra : dict, optional
        Extra sweeper parameters, e.g. ``level_precision``.
    multilevel : bool, optional
        Add a coarse level, which a DMDA injection coarsens by a factor of two.
    base_transfer_class : type, optional
        Space-time transfer, defaulting to pySDC's :class:`BaseTransfer`.
    maxiter : int, optional
        Number of SDC iterations.
    dt : float, optional
        Step size.
    nsteps : int, optional
        Number of steps.

    Returns
    -------
    tuple
        The end value and the problem instance.
    """
    description = {
        'problem_class': problem_class,
        'problem_params': problem_params,
        'sweeper_class': sweeper_class,
        'sweeper_params': dict(SWEEPER_PARAMS, **(sweeper_extra or {})),
        'level_params': {'restol': -1, 'dt': dt},
        'step_params': {'maxiter': maxiter},
    }
    if multilevel:
        description['problem_params'] = dict(problem_params, nvars=[PROBLEM_PARAMS['nvars'], COARSE_NVARS])
        description['space_transfer_class'] = mesh_to_mesh_petsc_dmda
        description['space_transfer_params'] = {'iorder': 2, 'rorder': 2, 'periodic': False}
        if base_transfer_class is not None:
            description['base_transfer_class'] = base_transfer_class

    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    prob = controller.MS[0].levels[0].prob
    uend, _ = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=nsteps * dt)
    return uend, prob


def configurations():
    """
    The comparison matrix, single- and multi-level, at full and emulated reduced precision.

    Returns
    -------
    list
        ``(label, problem_class, problem_extra, sweeper_class, sweeper_extra, run_kwargs)``.
    """
    delta_ml = {'multilevel': True, 'base_transfer_class': delta_transfer}
    f32, f16 = np.float32, np.float16
    return [
        ('SDC', petsc_fisher_fullyimplicit, {}, generic_implicit, {}, {}),
        ('deltaSDC', petsc_fisher_delta, {}, delta_implicit, {}, {}),
        ('fp32-deltaSDC', petsc_fisher_delta, {'solve_precision': f32}, delta_implicit, {}, {}),
        ('MLSDC', petsc_fisher_fullyimplicit, {}, generic_implicit, {}, {'multilevel': True}),
        ('deltaMLSDC', petsc_fisher_delta, {}, delta_implicit_rounded, {}, delta_ml),
        ('fp32-deltaMLSDC', petsc_fisher_delta, {'solve_precision': f32}, delta_implicit_rounded, {}, delta_ml),
        (
            'fp16-coarse-deltaMLSDC',
            petsc_fisher_delta,
            {'solve_precision': [f32, f16]},
            delta_implicit_rounded,
            {'level_precision': [None, f16]},
            delta_ml,
        ),
        (
            'cascade fp16>fp32>fp64 fine',
            petsc_fisher_delta,
            {'solve_precision': [f32, f16]},
            delta_implicit_cascade,
            {'level_precision': [None, f16], 'state_cascade': ('float16', 'float32', None)},
            delta_ml,
        ),
        # The controls. The first says the delta *hierarchy* is what makes a reduced-precision
        # coarse level safe, not the node-local solve; the second says precision on the fine level
        # binds, which is what makes every row above mean something.
        (
            'CONTROL fp16 coarse, stock ML',
            petsc_fisher_delta,
            {'solve_precision': [f32, f16]},
            delta_implicit_rounded,
            {'level_precision': [None, f16]},
            {'multilevel': True, 'base_transfer_class': rounding_transfer},
        ),
        (
            'CONTROL fp32 fine level',
            petsc_fisher_delta,
            {},
            delta_implicit_rounded,
            {'level_precision': f32},
            delta_ml,
        ),
    ]


def main():
    """
    Run the comparison matrix on the PETSc backend and check it.

    Returns
    -------
    dict
        The end value per configuration, keyed by label.

    Raises
    ------
    AssertionError
        If a configuration deviates from its own full-precision counterpart, or if a control fails
        to break.
    """
    results = {}
    for label, problem_class, problem_extra, sweeper_class, sweeper_extra, run_kwargs in configurations():
        results[label], prob = run(
            problem_class,
            dict(PROBLEM_PARAMS, **problem_extra),
            sweeper_class,
            sweeper_extra,
            **run_kwargs,
        )

    print(f"{'configuration':>30} | {'diff to SDC':>12} {'to fp64 peer':>13}")
    print('-' * 60)
    for label, uend in results.items():
        peer = results['MLSDC' if 'ML' in label else 'SDC']
        print(f'{label:>30} | {abs(uend - results["SDC"]):>12.3e} {abs(uend - peer):>13.3e}')

    assert prob.solve_precision is None or prob.solve_precision == np.dtype('float32')
    for label, uend in results.items():
        peer = results['MLSDC' if 'ML' in label else 'SDC']
        deviation = abs(uend - peer)
        if label.startswith('CONTROL'):
            assert deviation > 1e-9, f'{label} did not break, so it no longer controls anything'
        else:
            assert deviation < 1e-6, f'{label} deviates from its full-precision peer by {deviation:.3e}'

    return results


if __name__ == '__main__':
    main()
