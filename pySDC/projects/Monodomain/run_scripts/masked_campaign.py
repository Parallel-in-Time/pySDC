"""
Runs the masked-ESDC vs RL2 timing campaign on one node: every configuration is a single-threaded process pinned to
its own core, at most NPAR at a time, one per core complex (cores 0, STRIDE, 2*STRIDE, ...), so that runs do not share
an L3 cache and each socket's memory channels serve only a few of them.

    python masked_campaign.py OUT [NPAR=16] [STRIDE=8] [Tend=50] [SET=main]

SET=main is the 2D-slab work-precision campaign. SET=overnight writes three sub-campaigns into OUT/space, OUT/beat and
OUT/3d: the spatial error of the activation times (RL2 on refinements 0-2), a full beat with repolarisation on the
coarse 2D slab (400 ms), and the 3D Niederer slab (activation only); each subfolder is analysed on its own
(masked_spatial.py for space, masked_analyze.py with T=400 for beat and T=50 for 3d).

The runs are serial, but pySDC's mesh imports mpi4py, which would initialise MPI in every child. Under srun all children
inherit the launcher's PMIx identity and block in MPI_Init, so MPI initialisation is switched off for them
(MPI4PY_RC_INITIALIZE=0; nothing in a serial run calls MPI, and results are bit-identical).

Every run writes OUT/TAG.npz and appends to OUT/runs.jsonl (see masked_timing.py); its log goes to OUT/TAG.log.
Runs whose TAG.npz exists are skipped, so the campaign can be resumed.
"""

import os
import subprocess
import sys
import time
from pathlib import Path


def configs():
    """(tag, method, dt, extra args), longest first. Sized for one JUSUF node (2x EPYC 7742, 16 at a time) in < 3 h."""
    # reference: residual-masked ESDC converges to the same collocation solution as the unmasked one
    yield 'REF', 'ESDC_masked', 0.05, ['mask=res', 'restol=1e-8']
    yield 'RL2_0.00625', 'RL2', 0.00625, []
    for tol in ['1e-7', '1e-6']:
        yield f'masked_res_dt0.1_tol{tol}', 'ESDC_masked', 0.1, ['mask=res', f'restol={tol}']
    yield 'masked_res_dt0.05_tol1e-5', 'ESDC_masked', 0.05, ['mask=res', 'restol=1e-5']
    yield 'RL2_0.0125', 'RL2', 0.0125, []
    for dt in [0.1, 0.2]:
        yield f'masked_none_dt{dt:g}_tol1e-5', 'ESDC_masked', dt, ['mask=none', 'restol=1e-5']
    yield 'par_dt0.1_tol1e-5', 'ESDC_par', 0.1, ['restol=1e-5']
    for dt in [0.1, 0.2]:
        for kappa in ['0', '1']:
            yield f'masked_err_k{kappa}_dt{dt:g}_tol1e-5', 'ESDC_masked', dt, ['mask=err', 'theta=1', f'kappa={kappa}', 'restol=1e-5']
    for dt in [0.1, 0.2]:
        for tol in ['1e-6', '1e-5', '1e-4']:
            yield f'masked_inc_dt{dt:g}_tol{tol}', 'ESDC_masked', dt, ['mask=inc', f'restol={tol}']
    yield 'masked_res_dt0.2_tol1e-7', 'ESDC_masked', 0.2, ['mask=res', 'restol=1e-7']
    for dt in [0.1, 0.2]:
        for tol in ['1e-5', '1e-4', '1e-3']:
            yield f'masked_res_dt{dt:g}_tol{tol}', 'ESDC_masked', dt, ['mask=res', f'restol={tol}']
    yield 'masked_res_dt0.2_tol1e-6', 'ESDC_masked', 0.2, ['mask=res', 'restol=1e-6']
    for dt in [0.025, 0.05, 0.1]:
        yield f'RL2_{dt:g}', 'RL2', dt, []
    # timing controls: the same configurations again, late in the campaign
    yield 'RL2_0.0125_repeat', 'RL2', 0.0125, []
    yield 'masked_res_dt0.1_tol1e-5_repeat', 'ESDC_masked', 0.1, ['mask=res', 'restol=1e-5']


def overnight():
    """(tag, method, dt, extra args, subfolder), longest first."""
    beat = ['domain=cuboid_2D', 'refinements=0', 'Tend=400']
    d3 = ['domain=cuboid_3D', 'refinements=0', 'Tend=50']
    yield 'REF', 'ESDC_masked', 0.1, ['mask=res', 'restol=1e-8'] + d3, '3d'
    yield 'REF', 'ESDC_masked', 0.1, ['mask=res', 'restol=1e-8'] + beat, 'beat'
    yield 'space_ref2', 'RL2', 0.003125, ['refinements=2'], 'space'
    for dt, tol in [(0.1, '1e-5'), (0.2, '1e-5'), (0.2, '1e-4')]:
        yield f'masked_res_dt{dt:g}_tol{tol}', 'ESDC_masked', dt, ['mask=res', f'restol={tol}'] + d3, '3d'
    yield 'masked_none_dt0.1_tol1e-5', 'ESDC_masked', 0.1, ['mask=none', 'restol=1e-5'] + beat, 'beat'
    for dt in [0.1, 0.2]:
        for tol in ['1e-5', '1e-4']:
            yield f'masked_res_dt{dt:g}_tol{tol}', 'ESDC_masked', dt, ['mask=res', f'restol={tol}'] + beat, 'beat'
    for dt in [0.0125, 0.025, 0.05]:
        yield f'RL2_{dt:g}', 'RL2', dt, d3, '3d'
        yield f'RL2_{dt:g}', 'RL2', dt, beat, 'beat'
    yield 'space_ref1', 'RL2', 0.003125, ['refinements=1'], 'space'
    yield 'space_ref0', 'RL2', 0.003125, ['refinements=0'], 'space'


def main():
    out = Path(sys.argv[1])
    npar = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    Tend = sys.argv[4] if len(sys.argv) > 4 else '50'
    which = overnight if len(sys.argv) > 5 and sys.argv[5] == 'overnight' else (lambda: (c + ('',) for c in configs()))
    out.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name('masked_timing.py')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MPI4PY_RC_INITIALIZE='0')
    todo = [c for c in which() if not (out / c[4] / f'{c[0]}.npz').exists()]
    free = [i * stride for i in range(npar)]
    running = {}
    while todo or running:
        while todo and free:
            tag, method, dt, extra, sub = todo.pop(0)
            core = free.pop(0)
            (out / sub).mkdir(parents=True, exist_ok=True)
            cmd = ['taskset', '-c', str(core), sys.executable, str(script), str(out / sub), tag, method, str(dt), f'Tend={Tend}']
            cmd += extra  # a Tend in extra overrides the one above
            log = open(out / sub / f'{tag}.log', 'w')
            running[subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)] = (f'{sub}/{tag}', core, log)
            print(f'{time.strftime("%H:%M:%S")} start {sub}/{tag} on core {core}', flush=True)
        time.sleep(2)
        for p in [p for p in running if p.poll() is not None]:
            tag, core, log = running.pop(p)
            log.close()
            free.append(core)
            print(f'{time.strftime("%H:%M:%S")} done  {tag} (exit {p.returncode})', flush=True)


if __name__ == '__main__':
    main()
