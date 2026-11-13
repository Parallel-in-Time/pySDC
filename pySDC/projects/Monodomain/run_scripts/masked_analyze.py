"""
Work-precision table and plot for masked_timing.py runs: wall time against activation-time error w.r.t. the run tagged
REF, over the cells the reference activates before T - 5 ms (a run that misses one of them gets an infinite error), and
the same for repolarisation times where the reference has them.

    python masked_analyze.py OUT_DIR [T=50]
"""

import json
import sys
from pathlib import Path

import numpy as np


def main():
    d = Path(sys.argv[1])
    T = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
    rows = [json.loads(l) for l in open(d / 'runs.jsonl')]
    REF = np.load(d / 'REF.npz')
    ref = REF['tact']
    inner = np.isfinite(ref) & (ref < T - 5)
    print(f'compared {inner.sum()} of {inner.size} cells; reference leaves {np.isnan(ref).sum()} unactivated by T')
    # repolarisation times, where the reference has them (full-beat runs)
    rref = REF['trep'] if 'trep' in REF.files else np.full_like(ref, np.nan)
    rinner = np.isfinite(rref) & (rref < T - 5)
    if rinner.any():
        print(f'repolarisation compared on {rinner.sum()} cells')
    for r in rows:
        run = np.load(d / f'{r["tag"]}.npz')
        e = np.abs(run['tact'][inner] - ref[inner])
        r['emax'], r['emean'] = (np.inf, np.inf) if np.any(np.isnan(e)) else (float(e.max()), float(e.mean()))
        r['rmax'] = np.nan
        if rinner.any():
            er = np.abs(run['trep'][rinner] - rref[rinner])
            r['rmax'] = np.inf if np.any(np.isnan(er)) else float(er.max())
    rows = [r for r in rows if r['tag'] != 'REF']
    key = lambda r: (r['method'] + (('_' + r['mask']) if r['method'] == 'ESDC_masked' else ''), r['wall'])
    print(f'{"tag":34s} {"wall s":>8} {"max err ms":>10} {"mean err":>9} {"repol max":>9} {"sweeps":>6} {"active":>6}')
    for r in sorted(rows, key=key):
        print(f'{r["tag"]:34s} {r["wall"]:8.1f} {r["emax"]:10.2e} {r["emean"]:9.2e} {r["rmax"]:9.2e} {r["niter"]:6.2f} {r["active"]:6.3f}')

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    groups = {}
    for r in rows:
        groups.setdefault(key(r)[0], []).append(r)
    for g, rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r['emax'])
        ax.loglog([r['emax'] for r in rs], [r['wall'] for r in rs], 'o-', label=g)
    ax.set_xlabel('max activation-time error [ms]')
    ax.set_ylabel('wall time, 1 core [s]')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(d / 'work_precision.png', dpi=150)
    print('wrote', d / 'work_precision.png')


if __name__ == '__main__':
    main()
