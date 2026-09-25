"""
pySDC's GPU distributed FFT, checked against the released `mpi4py-fft` it replaces.

`pySDC.helpers.fft` exists because released `mpi4py-fft` keeps its data in NumPy arrays and
transposes with `MPI_Alltoallw`, so it cannot drive a GPU. That makes the CPU version the natural
oracle: the same decomposition, the same data, and the answers have to agree. These check the two
halves separately, so a failure says which one broke -- the spectral tests exercise them together
but cannot tell a bad transform from a bad transpose.
"""

import numpy as np
import pytest


@pytest.mark.cupy
def test_serial_transform_matches_cpu():
    """Our CuPyFFT against mpi4py-fft's serial FFT on the host, same data."""
    import cupy as cp
    from mpi4py_fft.libfft import FFT

    from pySDC.helpers.fft._serial import CuPyFFT

    shape, axes = (8, 6), (0, 1)
    rng = np.random.default_rng(0)
    u = rng.random(shape) + 1j * rng.random(shape)

    cpu = FFT(shape, axes, dtype=np.complex128, backend='scipy')
    gpu = CuPyFFT(shape, axes, dtype=np.complex128)

    got = cp.asnumpy(gpu.forward(cp.asarray(u)))
    expect = cpu.forward(u.copy())
    assert np.allclose(got, expect, rtol=0, atol=1e-12), f'forward differs by {abs(got - expect).max():.2e}'

    back = cp.asnumpy(gpu.backward(cp.asarray(got)))
    assert np.allclose(back, u, rtol=0, atol=1e-12), f'round trip differs by {abs(back - u).max():.2e}'


@pytest.mark.cupy
@pytest.mark.parallel(2)
def test_redistribute_matches_cpu():
    """Our NCCL transpose against mpi4py-fft's MPI one, same decomposition and data."""
    import cupy as cp
    from mpi4py import MPI
    from mpi4py_fft.distarray import DistArray

    from pySDC.helpers.fft._distarray import DistArrayCuPy

    shape = (8, 6)
    rng = np.random.default_rng(0)
    whole = rng.random(shape)

    cpu = DistArray(shape, dtype=float, alignment=0)
    gpu = DistArrayCuPy(shape, dtype=float, alignment=0)
    assert cpu.shape == gpu.shape, f'different local shapes: {cpu.shape} vs {gpu.shape}'
    assert cpu.local_slice() == gpu.local_slice(), 'different local slices'

    cpu[:] = whole[cpu.local_slice()]
    gpu[:] = cp.asarray(whole[gpu.local_slice()])

    cpu_moved = cpu.redistribute(1)
    gpu_moved = gpu.redistribute(1)

    assert cpu_moved.shape == gpu_moved.shape, f'{cpu_moved.shape} vs {gpu_moved.shape}'
    got, expect = cp.asnumpy(gpu_moved), np.asarray(cpu_moved)
    assert np.allclose(got, expect, rtol=0, atol=1e-14), f'transpose differs by {abs(got - expect).max():.2e}'
