"""
Work-precision comparison of exponential SDC against the Rush-Larsen schemes cardiac codes run.

Every run starts from rest, stimulates a corner and follows the front through the whole domain. The cost is counted
work (calls into the ionic model, DCT solves and Laplacians, field exponentials), weighted by the measured cost of
each kernel, so the comparison does not depend on the load of the machine. The quantity of interest is the
activation time (first upward crossing of ``V_THRESHOLD``) at every mesh point.
"""

import json
import sys
import time
import timeit
from pathlib import Path

import numpy as np

from pySDC.core.hooks import Hooks
from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.Monodomain.problem_classes.MonodomainODE import MultiscaleMonodomainODE
from pySDC.projects.Monodomain.sweeper_classes.exponential_runge_kutta.imexexp_1st_order import imexexp_1st_order
from pySDC.projects.Monodomain.sweeper_classes.runge_kutta.rush_larsen import rush_larsen

V_THRESHOLD = -20.0  # mV
DATA = Path(__file__).parent / '../../../../data/monodomain_work_precision'


def crossing_times(times, V):
    """
    First upward crossing of V_THRESHOLD between times[0] and times[-1], per column of V (shape len(times) x n).
    With two times this is linear interpolation; with more, the crossing of the interpolating polynomial.
    """
    n = V.shape[1]
    above = V >= V_THRESHOLD
    first = np.argmax(above, axis=0)
    tact = np.full(n, np.nan)
    new = above[-1] & ~above[0]
    if not np.any(new):
        return tact
    k = first[new]
    if len(times) == 2:
        v0, v1 = V[0, new], V[1, new]
        tact[new] = times[0] + (times[1] - times[0]) * (V_THRESHOLD - v0) / (v1 - v0)
        return tact
    # bisection on the interpolating polynomial inside the subinterval where the nodal values cross
    Vn = V[:, new]
    a, b = times[k - 1], times[k]
    weights = lambda t: np.array(
        [np.prod([(t - times[j]) / (times[i] - times[j]) for j in range(len(times)) if j != i], axis=0)
         for i in range(len(times))]
    )
    p = lambda t: np.sum(weights(t) * Vn, axis=0) - V_THRESHOLD
    pa = p(a)
    for _ in range(50):
        m = (a + b) / 2
        pm = p(m)
        left = np.sign(pm) == np.sign(pa)
        a, pa = np.where(left, m, a), np.where(left, pm, pa)
        b = np.where(left, b, m)
    tact[new] = (a + b) / 2
    return tact


class ActivationHook(Hooks):
    """Records activation times, using the collocation nodes as dense output where there are several"""

    label = 'run'

    def post_step(self, step, level_number):
        super().post_step(step, level_number)
        L = step.levels[0]
        P = L.prob
        if not hasattr(P, 'tact'):
            P.tact = np.full(L.u[0][0].shape, np.nan)
        nodes = L.sweep.coll.nodes
        times = L.time + L.dt * np.concatenate([[0.0], nodes])
        V = np.array([L.u[m][0] for m in range(len(times))])
        tact = crossing_times(times, V)
        new = np.isnan(P.tact) & np.isfinite(tact)
        P.tact[new] = tact[new]

        # a progress line about every 5% of simulated time, so long runs can be watched
        t, Tend = L.time + L.dt, P.Tend
        if not hasattr(self, 'next_report'):
            self.next_report, self.wall0, self.nsteps = 0.0, time.perf_counter(), 0
        self.nsteps += 1
        if t >= self.next_report or t >= Tend - 1e-12:
            print(
                f'[{self.label}] t = {t:7.2f} / {Tend:g} ms, {self.nsteps} steps, '
                f'{time.perf_counter() - self.wall0:6.0f} s wall',
                flush=True,
            )
            self.next_report = t + Tend / 20


def problem_params(domain='cube_2D', refinements=-2, model='TTP', Tend=200.0):
    return dict(
        order=4,
        refinements=refinements,
        domain_name=domain,
        ionic_model_name=model,
        stimulus='corner',
        read_init_val=False,
        init_time=0.0,
        init_val_name='',
        enable_output=False,
        output_V_only=True,
        output_root=str(DATA / 'tmp') + '/',
        output_file_name='unused',
        ref_sol='',
        end_time=Tend,
    )


def description(method, dt, M=4, sweeps=None, restol=1e-10, QI='IE', e_tol=None, dt_max=1.0, **kwargs):
    """
    method: 'RL1', 'RL2' (Strang DRD), 'RL2_RDR', 'ESDC' or 'ESDC_adaptive'. ESDC with a fixed number of sweeps if
    sweeps is given, else iterating to restol. ESDC_adaptive controls dt to the local tolerance e_tol, up to dt_max.
    """
    desc = dict(problem_class=MultiscaleMonodomainODE, problem_params=problem_params(**kwargs))
    if method.startswith('RL'):
        desc['sweeper_class'] = rush_larsen
        desc['sweeper_params'] = dict(
            scheme=method[:3], splitting='RDR' if method.endswith('RDR') else 'DRD', quad_type='RADAU-RIGHT',
            num_nodes=1,
        )
        desc['level_params'] = dict(dt=dt, restol=-1)
        desc['step_params'] = dict(maxiter=1)
    elif method.startswith('ESDC'):
        desc['sweeper_class'] = imexexp_1st_order
        desc['sweeper_params'] = dict(initial_guess='copy', quad_type='RADAU-RIGHT', num_nodes=M, QI=QI)
        if method == 'ESDC_adaptive':
            # dt is the initial step; the step-size controller needs a fixed number of sweeps
            from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

            desc['level_params'] = dict(dt=dt, restol=-1)
            desc['step_params'] = dict(maxiter=sweeps)
            desc['convergence_controllers'] = {Adaptivity: dict(e_tol=e_tol, dt_max=dt_max, dt_min=1e-4)}
        elif sweeps is None:
            desc['level_params'] = dict(dt=dt, restol=restol, residual_type='full_rel')
            desc['step_params'] = dict(maxiter=50)
        else:
            desc['level_params'] = dict(dt=dt, restol=-1)
            desc['step_params'] = dict(maxiter=sweeps)
    else:
        raise ValueError(method)
    return desc


def run(method, dt, label=None, **kwargs):
    desc = description(method, dt, **kwargs)
    Tend = desc['problem_params']['end_time']
    hook = type('LabelledActivationHook', (ActivationHook,), dict(label=label or f'{method} dt={dt:g}'))
    ctrl = controller_nonMPI(
        num_procs=1, controller_params=dict(logger_level=30, hook_class=[hook], mssdc_jac=False), description=desc
    )
    P = ctrl.MS[0].levels[0].prob
    wall = time.perf_counter()
    uend, stats = ctrl.run(u0=P.initial_value(), t0=0.0, Tend=Tend)
    wall = time.perf_counter() - wall
    niter = [v for _, v in get_sorted(stats, type='niter')]
    return dict(
        uend=np.array(uend[0]),
        tact=P.tact,
        wall=wall,
        niter=float(np.mean(niter)),
        counts={k: c.niter for k, c in P.work_counters.items()},
    )


def kernel_costs(**kwargs):
    """Measured cost of each counted kernel in milliseconds, on the problem the runs use"""
    P = MultiscaleMonodomainODE(**problem_params(**kwargs))
    u = P.initial_value()
    f = P.dtype_f(init=P.init)
    r = P.dtype_u(init=P.init)
    g = dict(P=P, u=u, f=f, r=r, np=np)
    t = lambda s: min(timeit.repeat(s, number=20, repeat=7, globals=g)) / 20 * 1e3
    return {
        'f_expl': t('P.eval_f_nonstiff(u, 0., f.expl)'),
        'lmbda_yinf': t('P.eval_lmbda_yinf_exp(u, P.lmbda, P.yinf)'),
        'lmbda': t('P.eval_lmbda_exp(u, P.lmbda)'),
        'solve': t('P.parabolic.solve_system(u[0], 0.1, u[0], 0., r[0])'),
        'laplacian': t('P.parabolic.add_disc_laplacian(u[0], r[0])'),
        'exp': t('np.exp(u[0])'),
    }


def main():
    """python work_precision.py METHOD DT [key=value ...]: one run, appended as a JSON line to DATA/runs.jsonl"""
    method, dt = sys.argv[1], float(sys.argv[2])
    kwargs = {}
    for arg in sys.argv[3:]:
        key, value = arg.split('=')
        kwargs[key] = value if key in ['domain', 'model', 'QI'] else float(value) if '.' in value or 'e' in value else int(value)
    res = run(method, dt, **kwargs)
    DATA.mkdir(parents=True, exist_ok=True)
    tag = '_'.join([method, f'{dt:g}'] + [f'{k}{v}' for k, v in sorted(kwargs.items())])
    np.savez(DATA / f'{tag}.npz', uend=res['uend'], tact=res['tact'])
    record = dict(method=method, dt=dt, **kwargs, wall=res['wall'], niter=res['niter'], counts=res['counts'], tag=tag)
    with open(DATA / 'runs.jsonl', 'a') as fh:
        fh.write(json.dumps(record) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
