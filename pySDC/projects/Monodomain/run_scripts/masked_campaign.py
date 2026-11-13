"""
Runs the masked-ESDC vs RL2 timing campaign on one node: every configuration is a single-threaded process pinned to
its own core, at most NPAR at a time, one per core complex (cores 0, STRIDE, 2*STRIDE, ...), so that runs do not share
an L3 cache and each socket's memory channels serve only a few of them.

    python masked_campaign.py OUT [NPAR=16] [STRIDE=8] [Tend=50] [SET=1]

SET 1 is the main campaign (it also computes REF). SET 2 adds the low-accuracy end (loose tolerances, large RL2 steps) and
the per-cell error-tolerance mode (mask=err); run it into the same OUT after SET 1.

Every run writes OUT/TAG.npz and appends to OUT/runs.jsonl (see masked_timing.py); its log goes to OUT/TAG.log.
Runs whose TAG.npz exists are skipped, so the campaign can be resumed.
"""

import os
import subprocess
import sys
import time
from pathlib import Path


def configs2():
    """Follow-up: loose tolerances, RL2 at large steps, mask=err (per-cell floor restol, theta=1)."""
    for dt in [0.1, 0.2]:
        for tol in ['1e-4', '1e-3']:
            yield f'masked_res_dt{dt:g}_tol{tol}', 'ESDC_masked', dt, ['mask=res', f'restol={tol}']
    for tol in ['1e-4', '1e-3']:
        yield f'masked_inc_dt0.2_tol{tol}', 'ESDC_masked', 0.2, ['mask=inc', f'restol={tol}']
    for dt in [0.1, 0.2]:
        for kappa in ['0', '1']:
            yield f'masked_err_k{kappa}_dt{dt:g}_tol1e-5', 'ESDC_masked', dt, ['mask=err', 'theta=1', f'kappa={kappa}', 'restol=1e-5']
    yield 'masked_err_k0_dt0.2_tol1e-4', 'ESDC_masked', 0.2, ['mask=err', 'theta=1', 'kappa=0', 'restol=1e-4']
    yield 'RL2_0.1', 'RL2', 0.1, []


def configs():
    """(tag, method, dt, extra args), longest first."""
    yield 'REF', 'ESDC_masked', 0.025, ['mask=none', 'restol=1e-9']
    for dt in [0.003125, 0.00625, 0.0125, 0.025, 0.05]:
        yield f'RL2_{dt:g}', 'RL2', dt, []
    for dt in [0.05, 0.1, 0.2]:
        for tol in ['1e-7', '1e-6', '1e-5']:
            for mask in ['res', 'inc']:
                yield f'masked_{mask}_dt{dt:g}_tol{tol}', 'ESDC_masked', dt, [f'mask={mask}', f'restol={tol}']
    for dt, tol in [(0.1, '1e-7'), (0.1, '1e-5'), (0.2, '1e-5')]:
        yield f'masked_none_dt{dt:g}_tol{tol}', 'ESDC_masked', dt, ['mask=none', f'restol={tol}']
    for dt, tol in [(0.1, '1e-6'), (0.2, '1e-5')]:
        yield f'masked_res_theta1_dt{dt:g}_tol{tol}', 'ESDC_masked', dt, ['mask=res', 'theta=1', f'restol={tol}']
    for dt in [0.1, 0.2]:
        yield f'par_dt{dt:g}_tol1e-5', 'ESDC_par', dt, ['restol=1e-5']
    # timing controls: the same configurations again, late in the campaign
    yield 'RL2_0.0125_repeat', 'RL2', 0.0125, []
    yield 'masked_res_dt0.1_tol1e-5_repeat', 'ESDC_masked', 0.1, ['mask=res', 'restol=1e-5']


def main():
    out = Path(sys.argv[1])
    npar = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    Tend = sys.argv[4] if len(sys.argv) > 4 else '50'
    which = configs2 if len(sys.argv) > 5 and sys.argv[5] == '2' else configs
    out.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name('masked_timing.py')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    todo = [c for c in which() if not (out / f'{c[0]}.npz').exists()]
    free = [i * stride for i in range(npar)]
    running = {}
    while todo or running:
        while todo and free:
            tag, method, dt, extra = todo.pop(0)
            core = free.pop(0)
            cmd = ['taskset', '-c', str(core), sys.executable, str(script), str(out), tag, method, str(dt), f'Tend={Tend}']
            cmd += extra
            log = open(out / f'{tag}.log', 'w')
            running[subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)] = (tag, core, log)
            print(f'{time.strftime("%H:%M:%S")} start {tag} on core {core}', flush=True)
        time.sleep(2)
        for p in [p for p in running if p.poll() is not None]:
            tag, core, log = running.pop(p)
            log.close()
            free.append(core)
            print(f'{time.strftime("%H:%M:%S")} done  {tag} (exit {p.returncode})', flush=True)


if __name__ == '__main__':
    main()
