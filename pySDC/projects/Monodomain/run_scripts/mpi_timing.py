"""
Wall-clock timing of RL2, serial ESDC and node-parallel ESDC (one collocation node per MPI rank).

    python mpi_timing.py METHOD DT OUTDIR [key=value ...]
    mpirun -n M python mpi_timing.py ESDC_par_MPI DT OUTDIR M=M [key=value ...]

METHOD: RL2, ESDC (serial sweep), ESDC_par (node-parallel sweep, all nodes on one rank), ESDC_par_MPI (one node per
rank). Keys: M, restol, domain, refinements, Tend. Rank 0 appends one JSON line per run to OUTDIR/timing.jsonl and saves
the final potential, so the MPI run can be checked against the same sweep on one rank.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.Monodomain.run_scripts.work_precision import description
from pySDC.projects.Monodomain.sweeper_classes.exponential_runge_kutta.imexexp_parallel_MPI import imexexp_parallel_MPI


def main():
    method, dt, out = sys.argv[1], float(sys.argv[2]), Path(sys.argv[3])
    kw = dict(domain='cube_2D', refinements=-2, Tend=250.0)
    for a in sys.argv[4:]:
        k, v = a.split('=')
        kw[k] = v if k == 'domain' else (float(v) if k in ['restol', 'Tend'] else int(v))
    comm = MPI.COMM_WORLD
    desc = description('ESDC_par' if method == 'ESDC_par_MPI' else method, dt, model='TTP', **kw)
    if method == 'ESDC_par_MPI':
        desc['sweeper_class'] = imexexp_parallel_MPI
        desc['sweeper_params']['comm'] = comm
    ctrl = controller_nonMPI(num_procs=1, controller_params=dict(logger_level=30, mssdc_jac=False), description=desc)
    P = ctrl.MS[0].levels[0].prob
    u0 = P.initial_value()
    comm.Barrier()
    wall = time.perf_counter()
    uend, stats = ctrl.run(u0=u0, t0=0.0, Tend=kw['Tend'])
    comm.Barrier()
    wall = time.perf_counter() - wall
    if comm.rank == 0:
        niter = [v for _, v in get_sorted(stats, type='niter')]
        tag = '_'.join([method, f'{dt:g}', f'np{comm.size}'] + [f'{k}{v}' for k, v in sorted(kw.items())])
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / f'{tag}.npy', np.asarray(uend[0]))
        rec = dict(tag=tag, method=method, dt=dt, ranks=comm.size, **kw, wall=wall, sweeps=float(np.mean(niter)),
                   steps=len(niter), counts={k: c.niter for k, c in P.work_counters.items()})
        with open(out / 'timing.jsonl', 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
        print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    main()
