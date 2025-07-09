import pytest
import numpy as np


@pytest.mark.parametrize("nZ", [1, 5, 16])
@pytest.mark.parametrize("nY", [1, 5, 16])
@pytest.mark.parametrize("nX", [1, 5, 16])
@pytest.mark.parametrize("nVar", [1, 2, 3])
def testVTR(nVar, nX, nY, nZ):
    from pySDC.helpers.vtkIO import writeToVTR, readFromVTR

    data1 = np.random.rand(nVar, nX, nY, nZ)
    coords1 = [np.sort(np.random.rand(n)) for n in [nX, nY, nZ]]
    varNames1 = [f"var{i}" for i in range(nVar)]

    data2, coords2, varNames2 = readFromVTR(writeToVTR("testVTR", data1, coords1, varNames1))

    for i, (x1, x2) in enumerate(zip(coords1, coords2)):
        print(x1, x2)
        assert np.allclose(x1, x2), f"coordinate mismatch in dir. {i}"
    assert varNames1 == varNames2, f"varNames mismatch"
    assert data1.shape == data2.shape, f"data shape mismatch"
    assert np.allclose(data1, data2), f"data values mismatch"
