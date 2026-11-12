import numpy as np
import pytest


@pytest.mark.cupy
@pytest.mark.parametrize('fmt', ['fp64', 'fp32', 'fp16'])
def test_fused_cg_is_the_unfused_cg(fmt):
    """On a GPU, CG with fused kernels computes bit for bit what the array operations compute."""
    import cupy as cp

    from pySDC.projects.DeltaSDC.krylov import DTYPES, cg

    n = 4096
    dtype = np.dtype(DTYPES[fmt])
    diagonal, c = dtype.type(1.4), dtype.type(0.2)

    def apply(x):
        return diagonal * x - c * (cp.roll(x, 1) + cp.roll(x, -1))

    b = cp.asarray(np.sin(2 * np.pi * np.arange(n) / n) + 0.1 * np.cos(14 * np.pi * np.arange(n) / n))
    rtol = 1e-2 if fmt == 'fp16' else 1e-10
    (x, k, reached), (y, m, _) = (cg(apply, b, fmt, rtol, 500, xp=cp, fused=fused) for fused in (True, False))
    assert k == m > 0 and cp.array_equal(x, y), (k, m)


@pytest.mark.cupy
def test_heat_cg_on_GPU():
    """The CG node solve of heat_cg on a GPU, on the fused stencil, agrees with the one on the CPU."""
    import cupy as cp

    from pySDC.projects.DeltaSDC.krylov import heat_cg

    params = {'nvars': 256, 'nu': 1.0, 'freq': 2, 'bc': 'periodic', 'cg_rtol': 1e-12}
    cpu, gpu = heat_cg(**params), heat_cg(useGPU=True, **params)
    x_cpu = cpu.solve_system(cpu.u_exact(0.0), 1e-3, cpu.u_exact(0.0), 0.0)
    x_gpu = gpu.solve_system(gpu.u_exact(0.0), 1e-3, gpu.u_exact(0.0), 0.0)
    assert np.abs(cp.asnumpy(x_gpu) - np.asarray(x_cpu)).max() < 1e-10
