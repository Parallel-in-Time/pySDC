import pytest


@pytest.mark.cupy
@pytest.mark.parametrize('imex', [False, True])
def test_GPU_does_not_leak(imex):
    """Building the problem on the GPU must leave later CPU instances, and the IMEX data type, alone."""
    import numpy as np
    import cupy as cp
    from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh
    from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh, imex_cupy_mesh
    from pySDC.implementations.problem_classes.polynomial_test_problem import (
        polynomial_testequation,
        polynomial_testequation_IMEX,
    )

    problem_class = polynomial_testequation_IMEX if imex else polynomial_testequation

    gpu = problem_class(useGPU=True)
    f = gpu.eval_f(gpu.u_exact(0.5), 0.5)
    assert gpu.xp is cp
    assert type(gpu.u_exact(0.5)) is cupy_mesh
    assert type(f) is (imex_cupy_mesh if imex else cupy_mesh)

    for cpu in [problem_class(), polynomial_testequation()]:
        assert cpu.xp is np
        assert type(cpu.u_exact(0.5)) is mesh
    assert type(problem_class().eval_f(problem_class().u_exact(0.5), 0.5)) is (imex_mesh if imex else mesh)
