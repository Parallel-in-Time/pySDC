import pytest
import numpy as np

from pySDC.helpers.fieldsIO import DTYPES, FieldsIO

FieldsIO.ALLOW_OVERWRITE = True


@pytest.mark.parametrize("dtypeIdx", DTYPES.keys())
@pytest.mark.parametrize("nDim", range(3))
def testHeader(nDim, dtypeIdx):
    from pySDC.helpers.fieldsIO import FieldsIO, Scal0D, Cart1D, Cart2D

    fileName = "testHeader.pysdc"
    dtype = DTYPES[dtypeIdx]

    coordX = np.linspace(0, 1, num=256, endpoint=False)
    coordY = np.linspace(0, 1, num=64, endpoint=False)

    if nDim == 0:
        Class = Scal0D
        args = {"nVar": 20}
    elif nDim == 1:
        Class = Cart1D
        args = {"nVar": 10, "coordX": coordX}
    elif nDim == 2:
        Class = Cart2D
        args = {"nVar": 10, "coordX": coordX, "coordY": coordY}

    f1 = Class(dtype, fileName)
    assert f1.__str__() == f1.__repr__(), "__repr__ and __str__ do not return the same result"
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


@pytest.mark.parametrize("dtypeIdx", DTYPES.keys())
@pytest.mark.parametrize("nSteps", [1, 2, 10, 100])
@pytest.mark.parametrize("nVar", [1, 2, 5])
def testScal0D(nVar, nSteps, dtypeIdx):
    from pySDC.helpers.fieldsIO import FieldsIO, Scal0D

    fileName = "testScal0D.pysdc"
    dtype = DTYPES[dtypeIdx]

    f1 = Scal0D(dtype, fileName)
    f1.setHeader(nVar=nVar)

    assert f1.nItems == nVar, f"{f1} do not have nItems == nVar"
    f1.initialize()

    u0 = np.random.rand(nVar).astype(f1.dtype)
    times = np.arange(nSteps) / nSteps

    for t in times:
        ut = (u0 * t).astype(f1.dtype)
        f1.addField(t, ut)

    assert f1.nFields == nSteps, f"{f1} do not have nFields == nSteps"
    assert np.allclose(f1.times, times), f"{f1} has wrong times stored in file"

    f2 = FieldsIO.fromFile(fileName)

    assert f1.nFields == f2.nFields, f"f2 ({f2}) has different nFields than f1 ({f1})"
    assert f1.times == f2.times, f"f2 ({f2}) has different times than f1 ({f1})"
    assert (f1.time(-1) == f2.times[-1]) and (
        f1.times[-1] == f2.time(-1)
    ), f"f2 ({f2}) has different last time than f1 ({f1})"

    for idx, t in enumerate(times):
        u1 = u0 * t
        t2, u2 = f2.readField(idx)
        assert t2 == t, f"{idx}'s fields in {f1} has incorrect time"
        assert u2.shape == u1.shape, f"{idx}'s fields in {f1} has incorrect shape"
        assert np.allclose(u2, u1), f"{idx}'s fields in {f1} has incorrect values"


@pytest.mark.parametrize("dtypeIdx", DTYPES.keys())
@pytest.mark.parametrize("nSteps", [1, 2, 5, 10])
@pytest.mark.parametrize("nX", [5, 10, 16, 32, 64])
@pytest.mark.parametrize("nVar", [1, 2, 5])
def testCart1D(nVar, nX, nSteps, dtypeIdx):
    from pySDC.helpers.fieldsIO import FieldsIO, Cart1D, DTYPES

    fileName = "testCart1D.pysdc"
    dtype = DTYPES[dtypeIdx]

    coordX = np.linspace(0, 1, num=nX, endpoint=False)
    nX = coordX.size

    f1 = Cart1D(dtype, fileName)
    f1.setHeader(nVar=nVar, coordX=coordX)

    assert f1.nItems == nVar * nX, f"{f1} do not have nItems == nVar*nX"
    assert f1.nX == nX, f"{f1} has incorrect nX"
    f1.initialize()

    u0 = np.random.rand(nVar, nX).astype(f1.dtype)
    times = np.arange(nSteps) / nSteps

    for t in times:
        ut = (u0 * t).astype(f1.dtype)
        f1.addField(t, ut)

    assert f1.nFields == nSteps, f"{f1} do not have nFields == nSteps"
    assert np.allclose(f1.times, times), f"{f1} has wrong times stored in file"

    f2 = FieldsIO.fromFile(fileName)

    assert f1.nFields == f2.nFields, f"f2 ({f2}) has different nFields than f1 ({f1})"
    assert f1.times == f2.times, f"f2 ({f2}) has different times than f1 ({f1})"
    assert (f1.time(-1) == f2.times[-1]) and (
        f1.times[-1] == f2.time(-1)
    ), f"f2 ({f2}) has different last time than f1 ({f1})"

    for idx, t in enumerate(times):
        u1 = u0 * t
        t2, u2 = f2.readField(idx)
        assert t2 == t, f"{idx}'s fields in {f1} has incorrect time"
        assert u2.shape == u1.shape, f"{idx}'s fields in {f1} has incorrect shape"
        assert np.allclose(u2, u1), f"{idx}'s fields in {f1} has incorrect values"


@pytest.mark.parametrize("dtypeIdx", DTYPES.keys())
@pytest.mark.parametrize("nSteps", [1, 2, 5, 10])
@pytest.mark.parametrize("nY", [5, 10, 16])
@pytest.mark.parametrize("nX", [5, 10, 16])
@pytest.mark.parametrize("nVar", [1, 2, 5])
def testCart2D(nVar, nX, nY, nSteps, dtypeIdx):
    from pySDC.helpers.fieldsIO import FieldsIO, Cart2D, DTYPES

    fileName = "testCart2D.pysdc"
    dtype = DTYPES[dtypeIdx]

    coordX = np.linspace(0, 1, num=nX, endpoint=False)
    coordY = np.linspace(0, 1, num=nY, endpoint=False)

    f1 = Cart2D(dtype, fileName)
    f1.setHeader(nVar=nVar, coordX=coordX, coordY=coordY)

    assert f1.nItems == nVar * nX * nY, f"{f1} do not have nItems == nVar*nX"
    assert f1.nX == nX, f"{f1} has incorrect nX"
    assert f1.nY == nY, f"{f1} has incorrect nY"
    f1.initialize()

    u0 = np.random.rand(nVar, nX, nY).astype(f1.dtype)
    times = np.arange(nSteps) / nSteps

    for t in times:
        ut = (u0 * t).astype(f1.dtype)
        f1.addField(t, ut)

    assert f1.nFields == nSteps, f"{f1} do not have nFields == nSteps"
    assert np.allclose(f1.times, times), f"{f1} has wrong times stored in file"

    f2 = FieldsIO.fromFile(fileName)

    assert f1.nFields == f2.nFields, f"f2 ({f2}) has different nFields than f1 ({f1})"
    assert f1.times == f2.times, f"f2 ({f2}) has different times than f1 ({f1})"
    assert (f1.time(-1) == f2.times[-1]) and (
        f1.times[-1] == f2.time(-1)
    ), f"f2 ({f2}) has different last time than f1 ({f1})"

    for idx, t in enumerate(times):
        u1 = u0 * t
        t2, u2 = f2.readField(idx)
        assert t2 == t, f"{idx}'s fields in {f1} has incorrect time"
        assert u2.shape == u1.shape, f"{idx}'s fields in {f1} has incorrect shape"
        assert np.allclose(u2, u1), f"{idx}'s fields in {f1} has incorrect values"


def initGrid(nVar, nX, nY=None):
    nDim = 1
    if nY is not None:
        nDim += 1
    x = np.linspace(0, 1, num=nX, endpoint=False)
    coords = (x,)
    gridSizes = (nX,)
    u0 = np.array(np.arange(nVar) + 1)[:, None] * x[None, :]

    if nDim > 1:
        y = np.linspace(0, 1, num=nY, endpoint=False)
        coords += (y,)
        gridSizes += (nY,)
        u0 = u0[:, :, None] * y[None, None, :]

    return coords, gridSizes, u0


def writeFields_MPI(fileName, nDim, dtypeIdx, algo, nSteps, nVar, nX, nY=None):
    coords, gridSizes, u0 = initGrid(nVar, nX, nY)

    from mpi4py import MPI
    from pySDC.helpers.blocks import BlockDecomposition
    from pySDC.helpers.fieldsIO import Cart1D, Cart2D

    comm = MPI.COMM_WORLD
    MPI_SIZE = comm.Get_size()
    MPI_RANK = comm.Get_rank()

    blocks = BlockDecomposition(MPI_SIZE, gridSizes, algo, MPI_RANK)

    if nDim == 1:
        (iLocX,), (nLocX,) = blocks.localBounds
        (pRankX,) = blocks.ranks
        Cart1D.setupMPI(comm, iLocX, nLocX)
        u0 = u0[:, iLocX : iLocX + nLocX]

        f1 = Cart1D(DTYPES[dtypeIdx], fileName)
        f1.setHeader(nVar=nVar, coordX=coords[0])

    if nDim == 2:
        (iLocX, iLocY), (nLocX, nLocY) = blocks.localBounds
        Cart2D.setupMPI(comm, iLocX, nLocX, iLocY, nLocY)
        u0 = u0[:, iLocX : iLocX + nLocX, iLocY : iLocY + nLocY]

        f1 = Cart2D(DTYPES[dtypeIdx], fileName)
        f1.setHeader(nVar=nVar, coordX=coords[0], coordY=coords[1])

    u0 = np.asarray(u0, dtype=f1.dtype)
    f1.initialize()

    times = np.arange(nSteps) / nSteps
    for t in times:
        ut = (u0 * t).astype(f1.dtype)
        f1.addField(t, ut)

    return u0


def compareFields_MPI(fileName, u0, nSteps):
    from pySDC.helpers.fieldsIO import FieldsIO

    f2 = FieldsIO.fromFile(fileName)

    times = np.arange(nSteps) / nSteps
    for idx, t in enumerate(times):
        u1 = u0 * t
        t2, u2 = f2.readField(idx)
        assert t2 == t, f"{idx}'s fields in {f2} has incorrect time"
        assert u2.shape == u1.shape, f"{idx}'s fields in {f2} has incorrect shape"
        assert np.allclose(u2, u1), f"{idx}'s fields in {f2} has incorrect values"


@pytest.mark.mpi4py
@pytest.mark.parametrize("nX", [61, 16, 32])
@pytest.mark.parametrize("nVar", [1, 4])
@pytest.mark.parametrize("nSteps", [1, 10])
@pytest.mark.parametrize("algo", ["ChatGPT", "Hybrid"])
@pytest.mark.parametrize("dtypeIdx", [0, 1])
@pytest.mark.parametrize("nProcs", [1, 2, 4])
def testCart1D_MPI(nProcs, dtypeIdx, algo, nSteps, nVar, nX):

    import subprocess

    fileName = "testCart1D_MPI.pysdc"

    cmd = f"mpirun -np {nProcs} python {__file__} --fileName {fileName} --nDim 1 "
    cmd += f"--dtypeIdx {dtypeIdx} --algo {algo} --nSteps {nSteps} --nVar {nVar} --nX {nX}"

    p = subprocess.Popen(cmd.split(), cwd=".")
    p.wait()
    assert p.returncode == 0, f"MPI write with {nProcs} did not return code 0, but {p.returncode}"

    from pySDC.helpers.fieldsIO import FieldsIO, Cart1D

    f2: Cart1D = FieldsIO.fromFile(fileName)

    assert type(f2) == Cart1D, f"incorrect type in MPI written fields {f2}"
    assert f2.nFields == nSteps, f"incorrect nFields in MPI written fields {f2}"
    assert f2.nVar == nVar, f"incorrect nVar in MPI written fields {f2}"
    assert f2.nX == nX, f"incorrect nX in MPI written fields {f2}"

    coords, _, u0 = initGrid(nVar, nX)
    assert np.allclose(f2.header['coordX'], coords[0]), f"incorrect coordX in MPI written fields {f2}"

    times = np.arange(nSteps) / nSteps
    for idx, t in enumerate(times):
        u1 = u0 * t
        t2, u2 = f2.readField(idx)
        assert t2 == t, f"{idx}'s fields in {f2} has incorrect time"
        assert u2.shape == u1.shape, f"{idx}'s fields in {f2} has incorrect shape"
        assert np.allclose(u2, u1), f"{idx}'s fields in {f2} has incorrect values"


@pytest.mark.mpi4py
@pytest.mark.parametrize("nY", [61, 16, 32])
@pytest.mark.parametrize("nX", [61, 16, 32])
@pytest.mark.parametrize("nVar", [1, 4])
@pytest.mark.parametrize("nSteps", [1, 10])
@pytest.mark.parametrize("algo", ["ChatGPT", "Hybrid"])
@pytest.mark.parametrize("dtypeIdx", [0, 1])
@pytest.mark.parametrize("nProcs", [1, 2, 4])
def testCart2D_MPI(nProcs, dtypeIdx, algo, nSteps, nVar, nX, nY):

    import subprocess

    fileName = "testCart2D_MPI.pysdc"

    cmd = f"mpirun -np {nProcs} python {__file__} --fileName {fileName} --nDim 2 "
    cmd += f"--dtypeIdx {dtypeIdx} --algo {algo} --nSteps {nSteps} --nVar {nVar} --nX {nX} --nY {nY}"

    p = subprocess.Popen(cmd.split(), cwd=".")
    p.wait()
    assert p.returncode == 0, f"MPI write with {nProcs} did not return code 0, but {p.returncode}"

    from pySDC.helpers.fieldsIO import FieldsIO, Cart2D

    f2: Cart2D = FieldsIO.fromFile(fileName)

    assert type(f2) == Cart2D, f"incorrect type in MPI written fields {f2}"
    assert f2.nFields == nSteps, f"incorrect nFields in MPI written fields {f2}"
    assert f2.nVar == nVar, f"incorrect nVar in MPI written fields {f2}"
    assert f2.nX == nX, f"incorrect nX in MPI written fields {f2}"
    assert f2.nY == nY, f"incorrect nY in MPI written fields {f2}"

    grids, _, u0 = initGrid(nVar, nX, nY)
    assert np.allclose(f2.header['coordX'], grids[0]), f"incorrect coordX in MPI written fields {f2}"
    assert np.allclose(f2.header['coordY'], grids[1]), f"incorrect coordY in MPI written fields {f2}"

    times = np.arange(nSteps) / nSteps
    for idx, t in enumerate(times):
        u1 = u0 * t
        t2, u2 = f2.readField(idx)
        assert t2 == t, f"{idx}'s fields in {f2} has incorrect time"
        assert u2.shape == u1.shape, f"{idx}'s fields in {f2} has incorrect shape"
        assert np.allclose(u2, u1), f"{idx}'s fields in {f2} has incorrect values"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--fileName', type=str, help='fileName of the file')
    parser.add_argument('--nDim', type=int, help='space dimension', choices=[1, 2])
    parser.add_argument('--dtypeIdx', type=int, help="dtype index", choices=DTYPES.keys())
    parser.add_argument(
        '--algo', type=str, help="algorithm used for block decomposition", choices=["ChatGPT", "Hybrid"]
    )
    parser.add_argument('--nSteps', type=int, help="number of field variables")
    parser.add_argument('--nVar', type=int, help="number of field variables")
    parser.add_argument('--nX', type=int, help="number of grid points in x dimension")
    parser.add_argument('--nY', type=int, help="number of grid points in y dimension")
    args = parser.parse_args()

    u0 = writeFields_MPI(**args.__dict__)
    compareFields_MPI(args.fileName, u0, args.nSteps)
