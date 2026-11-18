import pytest


@pytest.mark.petsc
@pytest.mark.parametrize('dof', [1, 2])
@pytest.mark.parametrize('sizes', [[7], [5, 4]])
@pytest.mark.parametrize('ghosted', [False, True])
def test_matches_getVecArray(dof, sizes, ghosted):
    """
    DMDAArray indexes a vector as DMDA.getVecArray does, also with ghost points, and its writes reach the vector.
    """
    import warnings
    import numpy as np
    from petsc4py import PETSc
    from pySDC.implementations.datatype_classes.petsc_vec import DMDAArray

    da = PETSc.DMDA().create(sizes, dof=dof, stencil_width=1, boundary_type=['periodic'] * len(sizes))
    vec = da.createLocalVec() if ghosted else da.createGlobalVec()
    vec.setArray(np.arange(vec.getLocalSize(), dtype=float))
    ranges = da.getGhostRanges() if ghosted else da.getRanges()
    index = tuple(slice(*r) for r in ranges)
    point = tuple(start for start, _ in ranges)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)  # what DMDAArray avoids
        expect = da.getVecArray(vec)[index].copy()

    array = DMDAArray(da, vec)
    assert np.array_equal(array[index], expect)
    assert np.array_equal(array[point], expect[(0,) * len(sizes)])
    assert np.array_equal(array[:], expect)

    norm = vec.norm(PETSc.NormType.INFINITY)
    array[point] = -1e3
    del array  # releases the buffer, which tells PETSc the vector changed
    assert vec.getArray()[0] == -1e3
    assert vec.norm(PETSc.NormType.INFINITY) == 1e3 != norm
