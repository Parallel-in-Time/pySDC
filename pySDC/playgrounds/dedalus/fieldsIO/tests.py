import pytest
import numpy as np


@pytest.mark.parametrize("dtypeIdx", range(6))
@pytest.mark.parametrize("nDim", range(3))
def testHeader(nDim, dtypeIdx):
    from base import FieldsIO, Scal0D, Cart1D, Cart2D, DTYPES

    fileName = "testHeader.pysdc"
    dtype = DTYPES[dtypeIdx]

    gridX = np.linspace(0, 1, num=256, endpoint=False)
    gridY = np.linspace(0, 1, num=64, endpoint=False)

    if nDim == 0:
        Class = Scal0D
        args = {"nVar": 20}
    elif nDim == 1:
        Class = Cart1D 
        args = {"nVar": 10, "gridX": gridX}
    elif nDim == 2:
        Class = Cart2D
        args = {"nVar": 10, "gridX": gridX, "gridY": gridY}

    f1 = Class(dtype, fileName)
    try:
        f1.initialize()
    except AssertionError:
        pass
    else:
        raise AssertionError(f"{f1} should not be initialized without AssertionError before header is set")

    f1.setHeader(**args)
    assert f1.header is not None, f"{f1} has still None for header after setHeader"
    assert f1.nItems is not None, f"{f1} has still None for nItems after setHeader"
    assert f1.nItems > 0, f"{f1} has nItems={f1.nItems} after setHeader" 
    try:
        f1.addField(0, np.zeros(f1.nItems, dtype=f1.dtype))
    except AssertionError:
        pass
    else:
        raise AssertionError(f"{f1} should not be initialized without error before header is set")

    f1.initialize()
    assert f1.initialized, f"{f1} is not initialized after calling initialize()"
    assert f1.fileSize == f1.hSize, f"{f1} has file size different than its header size"

    f2 = FieldsIO.fromFile(fileName)
    assert f2.initialized, f"f2 ({f2}) not initialized after instantiating from file"
    assert type(f2) == type(f1), f"f2 ({f2}) not of the same type as f1 ({f1})"
    assert f2.dtype == f1.dtype, f"f2 ({f2}) has not the same dtype as f1 ({f1})"
    
    for key, val in f1.header.items():
        assert key in f2.header, f"could not read {key} key in written {f2}"
        assert np.allclose(val, f2.header[key]), f"header's discrepancy for {key} in written {f2}"


@pytest.mark.parametrize("dtypeIdx", range(6))
@pytest.mark.parametrize("nSteps", [1, 2, 10, 100])
@pytest.mark.parametrize("nVar", [1, 2, 5])
def testScal0D(nVar, nSteps, dtypeIdx):
    from base import FieldsIO, Scal0D, DTYPES

    fileName = "testScal0D.pysdc"
    dtype = DTYPES[dtypeIdx]

    f1 = Scal0D(dtype, fileName)
    f1.setHeader(nVar=nVar)
    
    assert f1.nItems == nVar, f"{f1} do not have nItems == nVar"
    f1.initialize()

    u0 = np.random.rand(nVar).astype(f1.dtype)
    times = np.arange(nSteps)/nSteps

    for t in times:
        f1.addField(t, u0*t)

    assert f1.nFields == nSteps, f"{f1} do not have nFields == nSteps"
    assert np.allclose(f1.times, times), f"{f1} has wrong times stored in file"

    f2 = FieldsIO.fromFile(fileName)

    assert f1.nFields == f2.nFields, f"f2 ({f2}) has different nFields than f1 ({f1})"
    assert f1.times == f2.times, f"f2 ({f2}) has different times than f1 ({f1})"
    assert (f1.time(-1) == f2.times[-1]) and (f1.times[-1] == f2.time(-1)), \
        f"f2 ({f2}) has different last time than f1 ({f1})"
    
    for idx, t in enumerate(times):
        u1 = u0*t
        t2, u2 = f2.readField(idx)
        assert t2 == t, f"{idx}'s fields in {f1} has incorrect time"
        assert u2.shape == u1.shape, f"{idx}'s fields in {f1} has incorrect shape"
        assert np.allclose(u2, u1), f"{idx}'s fields in {f1} has incorrect values"


@pytest.mark.parametrize("dtypeIdx", range(6))
@pytest.mark.parametrize("nSteps", [1, 2, 5, 10])
@pytest.mark.parametrize("nX", [5, 10, 16, 32, 64])
@pytest.mark.parametrize("nVar", [1, 2, 5])
def testCart1D(nVar, nX, nSteps, dtypeIdx):
    from base import FieldsIO, Cart1D, DTYPES

    fileName = "testCart1D.pysdc"
    dtype = DTYPES[dtypeIdx]

    gridX = np.linspace(0, 1, num=nX, endpoint=False)
    nX = gridX.size

    f1 = Cart1D(dtype, fileName)
    f1.setHeader(nVar=nVar, gridX=gridX)
    
    assert f1.nItems == nVar*nX, f"{f1} do not have nItems == nVar*nX"
    assert f1.nX == nX, f"{f1} has incorrect nX"
    f1.initialize()

    u0 = np.random.rand(nVar, nX).astype(f1.dtype)
    times = np.arange(nSteps)/nSteps

    for t in times:
        f1.addField(t, u0*t)

    assert f1.nFields == nSteps, f"{f1} do not have nFields == nSteps"
    assert np.allclose(f1.times, times), f"{f1} has wrong times stored in file"

    f2 = FieldsIO.fromFile(fileName)

    assert f1.nFields == f2.nFields, f"f2 ({f2}) has different nFields than f1 ({f1})"
    assert f1.times == f2.times, f"f2 ({f2}) has different times than f1 ({f1})"
    assert (f1.time(-1) == f2.times[-1]) and (f1.times[-1] == f2.time(-1)), \
        f"f2 ({f2}) has different last time than f1 ({f1})"
    
    for idx, t in enumerate(times):
        u1 = u0*t
        t2, u2 = f2.readField(idx)
        assert t2 == t, f"{idx}'s fields in {f1} has incorrect time"
        assert u2.shape == u1.shape, f"{idx}'s fields in {f1} has incorrect shape"
        assert np.allclose(u2, u1), f"{idx}'s fields in {f1} has incorrect values"



@pytest.mark.parametrize("dtypeIdx", range(6))
@pytest.mark.parametrize("nSteps", [1, 2, 5, 10])
@pytest.mark.parametrize("nY", [5, 10, 16])
@pytest.mark.parametrize("nX", [5, 10, 16])
@pytest.mark.parametrize("nVar", [1, 2, 5])
def testCart2D(nVar, nX, nY, nSteps, dtypeIdx):
    from base import FieldsIO, Cart2D, DTYPES

    fileName = "testCart1D.pysdc"
    dtype = DTYPES[dtypeIdx]

    gridX = np.linspace(0, 1, num=nX, endpoint=False)
    gridY = np.linspace(0, 1, num=nY, endpoint=False)

    f1 = Cart2D(dtype, fileName)
    f1.setHeader(nVar=nVar, gridX=gridX, gridY=gridY)
    
    assert f1.nItems == nVar*nX*nY, f"{f1} do not have nItems == nVar*nX"
    assert f1.nX == nX, f"{f1} has incorrect nX"
    assert f1.nY == nY, f"{f1} has incorrect nY"
    f1.initialize()

    u0 = np.random.rand(nVar, nX, nY).astype(f1.dtype)
    times = np.arange(nSteps)/nSteps

    for t in times:
        f1.addField(t, u0*t)

    assert f1.nFields == nSteps, f"{f1} do not have nFields == nSteps"
    assert np.allclose(f1.times, times), f"{f1} has wrong times stored in file"

    f2 = FieldsIO.fromFile(fileName)

    assert f1.nFields == f2.nFields, f"f2 ({f2}) has different nFields than f1 ({f1})"
    assert f1.times == f2.times, f"f2 ({f2}) has different times than f1 ({f1})"
    assert (f1.time(-1) == f2.times[-1]) and (f1.times[-1] == f2.time(-1)), \
        f"f2 ({f2}) has different last time than f1 ({f1})"
    
    for idx, t in enumerate(times):
        u1 = u0*t
        t2, u2 = f2.readField(idx)
        assert t2 == t, f"{idx}'s fields in {f1} has incorrect time"
        assert u2.shape == u1.shape, f"{idx}'s fields in {f1} has incorrect shape"
        assert np.allclose(u2, u1), f"{idx}'s fields in {f1} has incorrect values"