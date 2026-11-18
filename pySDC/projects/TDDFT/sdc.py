r"""
SDC for the Kohn-Sham molecule: pySDC's IMEX sweeper with the kinetic part implicit, stopping on the residual.

``QI='IE', QE='EE'`` is the classical serial sweep. ``QI='MIN-SR-S', QE='PIC'`` makes every node update of a sweep
independent of the others (diagonal implicit part, explicit part from the previous iterate), so the M nodes can be
computed in parallel; its critical path is one node per sweep. The diagonal MIN-SR-S sweep grows the error
transiently on the oscillatory (imaginary) spectrum of the Schrödinger operator before it contracts, so a fixed small
number of sweeps can diverge where residual-based stopping simply iterates through the growth.
"""

import numpy as np

from pySDC.core.hooks import Hooks
from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.projects.TDDFT.problem import KohnShamMolecule1D


class RecordDipole(Hooks):
    """Record dipole and norm at multiples of the problem's output interval"""

    def post_step(self, step, level_number):
        super().post_step(step, level_number)
        L = step.levels[0]
        P = L.prob
        t = L.time + L.dt
        if abs(t / P.dt_out - round(t / P.dt_out)) < 1e-9:
            psi = np.asarray(L.uend)
            P.record['t'].append(t)
            P.record['d'].append(P.dipole(psi))
            P.record['n'].append(P.norm(psi))


def run_sdc(dt, Tend, dt_out, M=3, QI='IE', QE='EE', restol=1e-8, maxiter=40, problem_params=None):
    """
    Propagate the ground state with SDC on Gauss nodes, sweeping each step until the residual is below restol.

    Returns:
        dict: times ``t``, dipole ``d``, norm ``n``, ``cost`` (orbital FFT round trips on the critical path: per step
        norb (1 + k) for the parallel sweep, norb (1 + M k) for the serial one, with k the sweeps taken),
        ``cost_total`` and the mean and largest number of sweeps
    """
    desc = dict(
        problem_class=KohnShamMolecule1D,
        problem_params=problem_params or {},
        sweeper_class=imex_1st_order,
        sweeper_params=dict(quad_type='GAUSS', num_nodes=M, QI=QI, QE=QE, initial_guess='copy', do_coll_update=True),
        level_params=dict(dt=dt, restol=restol, residual_type='full_abs'),
        step_params=dict(maxiter=maxiter),
    )
    ctrl = controller_nonMPI(
        num_procs=1, controller_params=dict(logger_level=30, hook_class=[RecordDipole]), description=desc
    )
    P = ctrl.MS[0].levels[0].prob
    u0 = P.ground_state()
    P.dt_out = dt_out
    P.record = dict(t=[0.0], d=[P.dipole(np.asarray(u0))], n=[P.norm(np.asarray(u0))])
    _, stats = ctrl.run(u0=u0, t0=0.0, Tend=Tend)
    k = np.array([v for _, v in get_sorted(stats, type='niter')])
    total = int(P.norb * np.sum(1 + M * k))
    critical = int(P.norb * np.sum(1 + k)) if QI != 'IE' else total
    return dict(
        **{key: np.array(v) for key, v in P.record.items()},
        cost=critical,
        cost_total=total,
        sweeps_mean=float(k.mean()),
        sweeps_max=int(k.max()),
    )
