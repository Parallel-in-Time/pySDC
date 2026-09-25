"""
`mesh_to_mesh` on a GPU, checked against the same transfer on a CPU.

The operators are the same matrices either way, so the interesting question is not whether the
interpolation is right -- `test_mesh_to_mesh.py` settles that -- but whether a `cupy_mesh` reaches
them at all and comes back with the same numbers. It used to be rejected outright: the type checks
in `restrict` and `prolong` accepted only `'mesh'`.
"""

import numpy as np
import pytest


def transfer_both_ways(useGPU, nvars_fine=32, nvars_coarse=16, freq=2):
    """Restrict an exact solution and prolong it back, on one side of the PCI bus or the other."""
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh

    # `setup_GPU` switches the class rather than the instance, so the GPU side runs on a throwaway
    # subclass and leaves the shared class on the CPU for whatever runs next
    problem_class = type('heatNd_on_GPU', (heatNd_unforced,), {}) if useGPU else heatNd_unforced
    params = {'freq': freq, 'bc': 'periodic', 'useGPU': useGPU}

    fine = problem_class(nvars=nvars_fine, **params)
    coarse = problem_class(nvars=nvars_coarse, **params)
    transfer = mesh_to_mesh(fine, coarse, {'rorder': 2, 'iorder': 4, 'periodic': True})

    restricted = transfer.restrict(fine.u_exact(0.0))
    prolonged = transfer.prolong(restricted)

    as_numpy = (lambda a: a.get()) if useGPU else np.asarray
    return as_numpy(restricted), as_numpy(prolonged)


@pytest.mark.cupy
def test_mesh_to_mesh_on_GPU_matches_the_CPU():
    restricted_CPU, prolonged_CPU = transfer_both_ways(useGPU=False)
    restricted_GPU, prolonged_GPU = transfer_both_ways(useGPU=True)

    assert np.allclose(restricted_CPU, restricted_GPU, rtol=0, atol=1e-13), 'restriction differs from the CPU'
    assert np.allclose(prolonged_CPU, prolonged_GPU, rtol=0, atol=1e-13), 'prolongation differs from the CPU'

    # and the transfer has to have done something, or the comparison above is vacuous
    assert restricted_GPU.shape == (16,), f'restriction gave {restricted_GPU.shape}'
    assert prolonged_GPU.shape == (32,), f'prolongation gave {prolonged_GPU.shape}'
    assert abs(restricted_GPU).max() > 0.1, 'restriction returned near-zero, so nothing was transferred'
