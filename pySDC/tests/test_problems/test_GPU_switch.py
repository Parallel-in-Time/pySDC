"""
Switching one problem to the GPU must not switch every other problem of the same class.

`setup_GPU` swaps the array library, the sparse library and the datatypes. It used to do that on
the class, so a single `useGPU=True` instance reconfigured every instance in the process --
including ones built afterwards with `useGPU=False`, which then handed back CuPy arrays.
"""

import pytest


@pytest.mark.cupy
def test_one_GPU_instance_leaves_the_class_alone():
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced

    on_GPU = heatNd_unforced(nvars=32, freq=2, bc='periodic', useGPU=True)
    on_CPU = heatNd_unforced(nvars=32, freq=2, bc='periodic', useGPU=False)

    assert type(on_GPU.u_exact(0.0)).__name__ == 'cupy_mesh', 'the GPU instance is not on the GPU'
    assert type(on_CPU.u_exact(0.0)).__name__ == 'mesh', 'the GPU instance switched the CPU one with it'
    assert heatNd_unforced.dtype_u.__name__ == 'mesh', 'the class itself was switched'

    # and the order must not matter either
    again_on_GPU = heatNd_unforced(nvars=32, freq=2, bc='periodic', useGPU=True)
    assert type(again_on_GPU.u_exact(0.0)).__name__ == 'cupy_mesh', 'a later GPU instance was left on the CPU'
