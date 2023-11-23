from pySDC.implementations.problem_classes.TestEquation_0D import testequation0dXPU


class testequation0dGPU(testequation0dXPU):
    """
    GPU implementation of `testequation0dXPU`
    """

    from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh
    import cupy as xp
    import cupyx.scipy.sparse as xsp
    from cupyx.scipy.sparse.linalg import splu as _splu

    dtype_u = cupy_mesh
    dtype_f = cupy_mesh
    splu = staticmethod(_splu)
