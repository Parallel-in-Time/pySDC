"""
One timed run for the masked-ESDC vs Rush-Larsen comparison. Writes TAG.npz (activation times, final state) and appends
one JSON line (wall time, sweeps, work counters, mean active fraction) to OUT/runs.jsonl.

    python masked_timing.py OUT TAG METHOD DT [restol=1e-5] [mask=res] [theta=0.1] [domain=cuboid_2D] [refinements=1]
        [Tend=50] [M=4] [rest=eq]

rest=eq (default) starts every method from the cell model's equilibrated resting state, rest=paper from its published
initial values.

METHOD: RL2, ESDC_par (unmasked reference implementation) or ESDC_masked (with mask = none | res | inc | err;
err also takes kappa=0.1 and kmin=2).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.Monodomain.run_scripts import work_precision as wp
from pySDC.projects.Monodomain.sweeper_classes.exponential_runge_kutta.imexexp_masked import imexexp_masked


def resting_state(P, T=6e4):
    """
    The cell model's resting steady state: its published initial values are not one (some gates relax by 25% per
    0.1 ms step), and tissue simulations normally start from an equilibrated cell. Integrated for T ms without stimulus.
    """
    from scipy.integrate import solve_ivp

    def rhs(t, y):
        Y = np.ascontiguousarray(y.reshape(-1, 1))
        F = np.zeros_like(Y)
        P.ionic_model.f(Y, F)
        return F.ravel()

    y0 = np.array(P.ionic_model.initial_values(), dtype=float)
    sol = solve_ivp(rhs, (0.0, T), y0, method='BDF', rtol=1e-10, atol=1e-12)
    assert sol.success, sol.message
    return sol.y[:, -1]


def main():
    out, tag, method, dt = Path(sys.argv[1]), sys.argv[2], sys.argv[3], float(sys.argv[4])
    kw = dict(restol='1e-5', mask='res', theta='0.1', domain='cuboid_2D', refinements='1', Tend='50', M='4', rest='eq', kappa='0.1', kmin='2')
    kw.update(a.split('=', 1) for a in sys.argv[5:])
    prob = dict(domain=kw['domain'], refinements=int(kw['refinements']), model='TTP', Tend=float(kw['Tend']))
    if method == 'RL2':
        desc = wp.description('RL2', dt, **prob)
    else:
        desc = wp.description('ESDC_par', dt, M=int(kw['M']), restol=float(kw['restol']), **prob)
        if method == 'ESDC_masked':
            desc['sweeper_class'] = imexexp_masked
            desc['sweeper_params'].update(mask=kw['mask'], theta=float(kw['theta']), kappa=float(kw['kappa']), kmin=int(kw['kmin']))
    hook = type('Hook', (wp.ActivationHook,), dict(label=tag))
    ctrl = controller_nonMPI(
        num_procs=1, controller_params=dict(logger_level=30, hook_class=[hook], mssdc_jac=False), description=desc
    )
    P = ctrl.MS[0].levels[0].prob
    u0 = P.initial_value()
    if kw['rest'] == 'eq':
        np.asarray(u0)[:] = resting_state(P)[:, None]
    wall = time.perf_counter()
    uend, stats = ctrl.run(u0=u0, t0=0.0, Tend=prob['Tend'])
    wall = time.perf_counter() - wall
    niter = [v for _, v in wp.get_sorted(stats, type='niter')]
    sw = ctrl.MS[0].levels[0].sweep
    hist = np.array(getattr(sw, 'active_history', [])).reshape(-1, 2)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f'{tag}.npz', tact=P.tact, uend=np.array(uend[0]), active=hist)
    row = dict(tag=tag, method=method, dt=dt, **kw, wall=wall, niter=float(np.mean(niter)), ncells=int(P.parabolic.n_dofs),
               active=float(hist[:, 1].mean()) if len(hist) else 1.0,
               counts={k: c.niter for k, c in P.work_counters.items()})
    with open(out / 'runs.jsonl', 'a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
