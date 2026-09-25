r"""
Time to solution of the precision ladder on real hardware: 2D heat, CG solves, CPU or GPU.

Every other script in this project measures *whether* a reduced precision costs accuracy or
iterations. This one measures what it buys, so nothing here is emulated: a reduced-precision solve
really reads fp32 arrays (:class:`.problems.heat_solve_dtype`) and a reduced coarse level really
holds them (``dtype``). Every row runs to the same residual tolerance, and reports the answer's
distance from the fp64 SDC one next to its wall time.

On a GPU, from the repository root::

    modal run etc/modal_gpu_tests.py --script pySDC/projects/DeltaSDC/run_gpu.py

or with ``python run_gpu.py --gpu`` wherever CuPy is installed.
"""

import argparse
import time

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.delta_form import delta_implicit
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.transfer_classes.BaseTransferDelta import delta_transfer
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.projects.DeltaSDC.problems import heat_solve_dtype

SWEEPER_PARAMS = {
    'quad_type': 'RADAU-RIGHT',
    'node_type': 'LEGENDRE',
    'num_nodes': 3,
    'QI': 'LU',
    'initial_guess': 'spread',
}

RESTOL = 1e-10

TIGHT, LOOSE = 1e-12, 1e-5
"""CG tolerances. Stock SDC solves for the state, so its solve has to be as accurate as the answer;
the delta form solves for a correction and needs about five digits (see ``plot_delivered_accuracy``)."""


def configurations():
    """
    ``(label, sweeper_class, problem_params, sweeper_params, multilevel)``.

    Parameters given as lists are spread over the levels by pySDC, fine level first.
    """
    delta = {'linear_implicit': True}
    return [
        ('SDC, CG 1e-12', generic_implicit, {'lintol': TIGHT}, {}, False),
        ('CONTROL SDC, CG 1e-5', generic_implicit, {'lintol': LOOSE}, {}, False),
        ('deltaSDC, CG 1e-5', delta_implicit, {'lintol': LOOSE}, delta, False),
        ('deltaSDC, fp32 CG 1e-5', delta_implicit, {'lintol': LOOSE, 'solve_dtype': 'float32'}, delta, False),
        ('MLSDC, CG 1e-12', generic_implicit, {'lintol': TIGHT}, {}, True),
        ('deltaMLSDC, CG 1e-5', delta_implicit, {'lintol': LOOSE}, delta, True),
        (
            'deltaMLSDC, fp32 fine CG + fp32 coarse',
            delta_implicit,
            {'lintol': LOOSE, 'solve_dtype': ['float32', None], 'dtype': ['float64', 'float32']},
            delta,
            True,
        ),
        (
            # fp16 storage, fp32 arithmetic: neither SciPy nor CuPy has a half-precision sparse matrix
            'deltaMLSDC, fp32 fine CG + fp16 coarse',
            delta_implicit,
            {'lintol': LOOSE, 'solve_dtype': ['float32', None], 'dtype': ['float64', 'float16']},
            delta,
            True,
        ),
    ]


def fft_configurations():
    """
    The delivered-accuracy specification, with the solve done by FFT at up to half precision.

    The README's specification says what the iteration needs from a node-local solve: about four
    digits for SDC's, six for MLSDC's fine one, two for MLSDC's coarse one. A half-precision FFT
    delivers about three, genuinely on a GPU -- so SDC should lose an iteration or so, MLSDC's
    coarse solve nothing, and MLSDC's fine solve clearly more. Same row format as
    :func:`configurations`.
    """
    delta = {'linear_implicit': True}
    fft = {'solver_type': 'FFT'}
    return [
        ('SDC, FFT', generic_implicit, fft, {}, False),
        ('deltaSDC, fp32 FFT', delta_implicit, {**fft, 'solve_dtype': 'float32'}, delta, False),
        ('deltaSDC, fp16 FFT', delta_implicit, {**fft, 'solve_dtype': 'float16'}, delta, False),
        ('MLSDC, FFT', generic_implicit, fft, {}, True),
        ('deltaMLSDC, fp16 coarse FFT', delta_implicit, {**fft, 'solve_dtype': [None, 'float16']}, delta, True),
        ('deltaMLSDC, fp16 fine FFT', delta_implicit, {**fft, 'solve_dtype': ['float16', None]}, delta, True),
    ]


def run(n, sweeper_class, problem_params, sweeper_params, multilevel, use_gpu, dt, nsteps):
    """
    Build and run one configuration.

    Returns
    -------
    tuple
        End value, wall time in seconds, iterations of the last step, inner solver work on all levels
        (CG iterations, or FFT solves), and
        the controller, for inspecting the levels afterwards.
    """
    problem_params = {
        'nvars': (n, n),
        'nu': 0.1,
        'bc': 'periodic',
        'solver_type': 'CG',
        'liniter': 10000,
        'useGPU': use_gpu,
        **problem_params,
    }
    description = {
        'problem_class': heat_solve_dtype,
        'problem_params': problem_params,
        'sweeper_class': sweeper_class,
        'sweeper_params': dict(SWEEPER_PARAMS, **sweeper_params),
        'level_params': {'restol': RESTOL, 'dt': dt},
        'step_params': {'maxiter': 50},
    }
    if multilevel:
        problem_params['nvars'] = [(n, n), (n // 2, n // 2)]
        description['space_transfer_class'] = mesh_to_mesh
        description['space_transfer_params'] = {'iorder': 4, 'rorder': 2, 'periodic': True}
        if sweeper_class is delta_implicit:
            description['base_transfer_class'] = delta_transfer

    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    prob = controller.MS[0].levels[0].prob
    # not `u_exact`: its single Fourier mode is an eigenvector of the Laplacian, so CG converges in
    # one iteration in double and only rounding to fp32 adds the other modes -- a bias against fp32
    x, y = prob.grids
    u0 = prob.dtype_u(prob.init)
    u0[:] = prob.xp.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / (2 * 0.05**2))
    sync()
    start = time.perf_counter()
    uend, stats = controller.run(u0=u0, t0=0.0, Tend=nsteps * dt)
    sync()
    wall = time.perf_counter() - start
    niter = get_sorted(stats, type='niter')[-1][1]
    cg = sum(level.prob.work_counters[level.prob.solver_type].niter for level in controller.MS[0].levels)
    return uend, wall, niter, cg, controller


PARADIAG_RESTOL = 1e-10
"""Not tighter: the residual floor in double grows like dt ||A|| eps, about 1e-12 at 1024 x 1024."""

PARADIAG_ROWS = [
    ('ParaDiag, complex128', None, None),
    ('ParaDiag, complex64 solve', 'complex64', None),
    ('ParaDiag, complex64 transform', None, 'complex64'),
    ('ParaDiag, complex64 preconditioner', 'complex64', 'complex64'),
]
"""``(label, solve_precision, transform_precision)``. Everything inside the preconditioner may drop to
complex64; the solution, the residual and the update stay complex128 (see ``paradiag.py``)."""


def run_paradiag(n, solve_precision, transform_precision, use_gpu, dt, nsteps, alpha=1e-4):
    r"""
    One ParaDiag block over ``nsteps`` steps, 2D periodic heat, FFT node-local solves.

    ``alpha`` is a number, or ``'adaptive'`` for :class:`AdaptiveAlpha` with its accuracy floor
    :math:`\gamma = L(3\varepsilon + \tau)` told the precision the preconditioner actually runs at
    (as ``inner_tol``), or ``'adaptive-fp64'`` for the stock floor, which assumes double throughout.

    Returns
    -------
    tuple
        End value, wall time in seconds, iterations, and the residual reached.
    """
    from pySDC.implementations.sweeper_classes.ParaDiagSweepers import QDiagonalizationIMEX
    from pySDC.projects.DeltaSDC.paradiag import controller_ParaDiag_reduced_transform, heat_paradiag

    description = {
        'problem_class': heat_paradiag,
        'problem_params': {
            'nvars': (n, n),
            'nu': 0.1,
            'freq': (2, 2),
            'bc': 'periodic',
            'solve_precision': solve_precision,
            'useGPU': use_gpu,
        },
        'sweeper_class': QDiagonalizationIMEX,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3},
        'level_params': {'restol': PARADIAG_RESTOL, 'dt': dt},
        'step_params': {'maxiter': 30},
    }
    if str(alpha).startswith('adaptive'):
        import numpy as np

        from pySDC.implementations.convergence_controller_classes.adaptive_alpha import AdaptiveAlpha

        reduced = 'complex64' in (solve_precision, transform_precision) and alpha == 'adaptive'
        inner_tol = float(np.finfo(np.complex64).eps) if reduced else 0.0
        description['convergence_controllers'] = {AdaptiveAlpha: {'inner_tol': inner_tol}}
        alpha = 1e-4  # the first transform needs one; the controller takes over from there
    controller_params = {'logger_level': 40, 'alpha': float(alpha)}
    if transform_precision is not None:
        controller_params['transform_precision'] = transform_precision
    controller = controller_ParaDiag_reduced_transform(
        num_procs=nsteps, controller_params=controller_params, description=description
    )
    u0 = controller.MS[0].levels[0].prob.u_exact(0.0)
    sync()
    start = time.perf_counter()
    uend, stats = controller.run(u0=u0, t0=0.0, Tend=nsteps * dt)
    sync()
    wall = time.perf_counter() - start
    niter = max(value for _, value in get_sorted(stats, type='niter'))
    residual = min(value for _, value in get_sorted(stats, type='residual_post_iteration'))
    return uend, wall, niter, residual


def main_paradiag(sizes, use_gpu, dt, nsteps, alphas):
    """Print the ParaDiag precision table for each grid size."""
    for n, alpha in [(n, alpha) for n in sizes for alpha in alphas]:
        print(f'\nParaDiag {n}x{n}, alpha={alpha}, dt={dt}, {nsteps} steps in one block, restol={PARADIAG_RESTOL}')
        print(f"{'configuration':>40} | {'iter':>4} {'residual':>9} | {'wall [s]':>8} {'speedup':>7} | {'diff':>9}")
        print('-' * 90)
        reference = None
        for label, solve_precision, transform_precision in PARADIAG_ROWS:
            if use_gpu:
                run_paradiag(64, solve_precision, transform_precision, use_gpu, dt, nsteps, alpha)
            uend, wall, niter, residual = run_paradiag(
                n, solve_precision, transform_precision, use_gpu, dt, nsteps, alpha
            )
            if reference is None:
                reference = (uend, wall)
            diff = float(abs(uend - reference[0]))
            print(
                f'{label:>40} | {niter:>4} {residual:>9.2e} | {wall:>8.3f} {reference[1] / wall:>7.2f} | {diff:>9.2e}'
            )


def sync():
    """Wait for the GPU, if there is one, so that a timer measures the work and not its launch."""
    try:
        import cupy

        cupy.cuda.Device().synchronize()
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, nargs='+', default=[1024])
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--dt', type=float, default=1e-2)
    parser.add_argument('--nsteps', type=int, default=2)
    parser.add_argument('--only', nargs='+', default=None, help='run only rows whose label contains one of these')
    parser.add_argument('--paradiag', action='store_true', help='the ParaDiag table instead of SDC/MLSDC')
    parser.add_argument('--fft', action='store_true', help='FFT solves down to half precision, instead of CG')
    parser.add_argument(
        '--alpha', nargs='+', default=['1e-4'], help="ParaDiag alpha(s): numbers, 'adaptive', 'adaptive-fp64'"
    )
    args = parser.parse_args()
    use_gpu = args.gpu
    if not use_gpu:
        try:
            import cupy

            use_gpu = cupy.cuda.runtime.getDeviceCount() > 0
        except Exception:
            pass
    if use_gpu:
        import cupy

        print(f"device: {cupy.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    else:
        print('device: CPU')

    if args.paradiag:
        return main_paradiag(args.n, use_gpu, args.dt, args.nsteps, args.alpha)

    for n in args.n:
        print(f'\n{n}x{n}, dt={args.dt}, {args.nsteps} steps, restol={RESTOL}')
        print(
            f"{'configuration':>40} | {'iter':>4} {'inner':>6} | {'wall [s]':>8} {'speedup':>7} | {'diff to SDC':>11}"
        )
        print('-' * 90)
        reference = None
        for label, sweeper_class, problem_params, sweeper_params, multilevel in (
            fft_configurations() if args.fft else configurations()
        ):
            # the first row is the reference, so it always runs
            if reference is not None and args.only and not any(key in label for key in args.only):
                continue
            if use_gpu:
                # CuPy compiles kernels on first use, per dtype; pay for that on a small grid, untimed
                run(64, sweeper_class, problem_params, sweeper_params, multilevel, use_gpu, args.dt, 1)
            uend, wall, niter, cg, _ = run(
                n, sweeper_class, problem_params, sweeper_params, multilevel, use_gpu, args.dt, args.nsteps
            )
            if reference is None:
                reference = (uend, wall)
            diff = abs(uend - reference[0])
            print(f'{label:>40} | {niter:>4} {cg:>6} | {wall:>8.3f} {reference[1] / wall:>7.2f} | {diff:>11.2e}')


if __name__ == '__main__':
    main()
