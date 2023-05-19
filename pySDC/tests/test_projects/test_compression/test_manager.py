import pytest


# Test to compress an array of numbers and decompress it
@pytest.mark.libpressio
@pytest.mark.parametrize("shape_t", [(100,), (100, 100), (100, 100, 100)])
@pytest.mark.parametrize("errorBound", [1e-1, 1e-3])
def test_compression(shape_t, errorBound):
    from pySDC.projects.compression.compressed_mesh import compressed_mesh
    from pySDC.implementations.datatype_classes.mesh import mesh
    import numpy as np
    from pySDC.projects.compression.CRAM_Manager import CRAM_Manager

    np_rng = np.random.default_rng(seed=4)
    arr = np_rng.random(shape_t)

    dtype = compressed_mesh(init=(shape_t, None, np.float64))
    dtype2 = mesh(init=(shape_t, None, np.dtype("float64")))
    dtype2[:] = arr[:]
    dtype.manager.compress(arr[:], dtype.name, 0, errBound=errorBound)

    error = abs(dtype[:] - dtype2[:])
    assert (
        error > 0 or errorBound < 1e-1
    ), f"Compression did nothing(lossless compression), got error:{error:.2e} with error bound: {errorBound:.2e}"
    assert (
        error <= errorBound
    ), f"Error too large, compression failed, got error: {error:.2e} with error bound: {errorBound:.2e}"


def test_mesh_operations():
    from pySDC.projects.compression.compressed_mesh import compressed_mesh
    import numpy as np

    # TODO: Add method to change default error bound before creating first mesh
    arr1 = compressed_mesh(init=((30,), None, np.float64), val=1)
    arr2 = compressed_mesh(init=((30,), None, np.float64), val=2.0)

    np_arr1 = np.ones((30,)) * 3.0

    assert all(
        me == 3.0 for me in (arr1 + arr2)
    ), "Addition of two compressed meshes failed unexpectedly."
    assert all(
        me == -1 for me in (arr1 - arr2)
    ), "Subtraction of two compressed meshes failed unexpectedly."
    assert all(
        me == 4 for me in (arr1 + np_arr1)
    ), "Addition of a compressed mesh and numpy array failed unexpectedly."
    assert all(
        me == -2 for me in (arr1 - np_arr1)
    ), "Subtraction of a compressed mesh and numpy array failed unexpectedly."
    assert all(
        me == 5 for me in (4.0 + arr1)
    ), "Addition of a float and compressed mesh failed unexpectedly."
    assert all(
        me == 2 for me in (4.0 - arr2)
    ), "Subtraction of a float and compressed mesh failed unexpectedly."
    assert all(
        me == 4 for me in (4.0 * arr1)
    ), "Multiplication of a float and compressed mesh failed unexpectedly."
    assert all(
        me == 5 for me in (arr1 + 4.0)
    ), "Addition of a compressed mesh and float failed unexpectedly."
    assert all(
        me == -2 for me in (arr2 - 4.0)
    ), "Subtraction of a compressed mesh and float failed unexpectedly."


if __name__ == "__main__":
    test_mesh_operations()
