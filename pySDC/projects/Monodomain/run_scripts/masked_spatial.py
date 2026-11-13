"""
Spatial error of the activation times: runs of the same method and step on refinements 0, 1, 2 (tags space_ref0/1/2 in
OUT), each compared with the finest one interpolated (bilinear, cell-centred grids) onto its cell centres.

    python masked_spatial.py OUT [T=50]
"""

import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

DOM = (20.0, 7.0)  # cuboid_2D


def grid(ref):
    n = [int(2 ** np.round(np.log2(5.0 * L * 2**ref))) for L in DOM]
    return [(np.arange(k) + 0.5) * L / k for k, L in zip(n, DOM)]


def main():
    d = Path(sys.argv[1])
    T = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
    xf = grid(2)
    tf = np.load(d / 'space_ref2.npz')['tact'].reshape(len(xf[1]), len(xf[0]))  # (ny, nx): the DCT grid's order
    # extrapolate a half cell at the boundary: the coarse cell centres lie outside the fine ones' hull
    fine = RegularGridInterpolator((xf[1], xf[0]), tf, bounds_error=False, fill_value=None)
    for ref in [0, 1]:
        x = grid(ref)
        tc = np.load(d / f'space_ref{ref}.npz')['tact']
        Y, X = np.meshgrid(x[1], x[0], indexing='ij')
        tfi = fine(np.stack([Y.ravel(), X.ravel()], axis=1))
        ok = np.isfinite(tfi) & (tfi < T - 5)
        e = np.abs(tc[ok] - tfi[ok])
        h = DOM[0] / len(x[0])
        print(f'refinement {ref} (dx = {h:.3f} mm): activation-time error vs refinement 2: max {np.nanmax(e):.3f} ms, '
              f'mean {np.nanmean(e):.3f} ms, over {ok.sum()} cells, {np.isnan(tc[ok]).sum()} missed')


if __name__ == '__main__':
    main()
