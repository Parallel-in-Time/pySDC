import os
import sys
import glob
import pytest

if sys.version_info < (3, 11):
    pytest.skip("skipping fieldsIO tests on python lower than 3.11", allow_module_level=True)

import itertools
import numpy as np

from pySDC.helpers.fieldsIO import DTYPES, FieldsIO

FieldsIO.ALLOW_OVERWRITE = True


@pytest.mark.parametrize("dtypeIdx", DTYPES.keys())
@pytest.mark.parametrize("dim", range(4))
def testHeader(dim, dtypeIdx):
    from pySDC.helpers.fieldsIO import FieldsIO, Scalar, Rectilinear

    fileName = "testHeader.pysdc"
    dtype = DTYPES[dtypeIdx]

    coords = [np.linspace(0, 1, num=256, endpoint=False) for n in [256, 64, 32]]

    if dim == 0:
        Class = Scalar
        args = {"nVar": 20}
    else:
        Class = Rectilinear
        args = {"nVar": 10, "coords": coords[:dim]}

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
def testScalar(nVar, nSteps, dtypeIdx):
    from pySDC.helpers.fieldsIO import FieldsIO, Scalar

    fileName = "testScalar.pysdc"
    dtype = DTYPES[dtypeIdx]

    f1 = Scalar(dtype, fileName)
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
@pytest.mark.parametrize("nVar", [1, 2, 5])
@pytest.mark.parametrize("dim", [1, 2, 3])
def testRectilinear(dim, nVar, nSteps, dtypeIdx):
    from pySDC.helpers.fieldsIO import FieldsIO, Rectilinear, DTYPES

    fileName = f"testRectilinear{dim}D.pysdc"
    dtype = DTYPES[dtypeIdx]

    for gridSizes in itertools.product(*[[5, 10, 16]] * dim):

        coords = [np.linspace(0, 1, num=n, endpoint=False) for n in gridSizes]

        f1 = Rectilinear(dtype, fileName)
        f1.setHeader(nVar=nVar, coords=coords)

        assert f1.dim == dim, f"{f1} has incorrect dimension"
        assert f1.gridSizes == list(gridSizes), f"{f1} has incorrect gridSizes"
        assert f1.nDoF == np.prod(gridSizes), f"{f1} has incorrect nDOF"
        assert f1.nItems == nVar * np.prod(gridSizes), f"{f1} do not have nItems == nVar*product(gridSizes)"

        f1.initialize()
        u0 = np.random.rand(nVar, *gridSizes).astype(f1.dtype)
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


@pytest.mark.parametrize("nSteps", [1, 10])
@pytest.mark.parametrize("nZ", [1, 5, 16])
@pytest.mark.parametrize("nY", [1, 5, 16])
@pytest.mark.parametrize("nX", [1, 5, 16])
@pytest.mark.parametrize("nVar", [1, 2, 3])
def testToVTR(nVar, nX, nY, nZ, nSteps):

    from pySDC.helpers.fieldsIO import Rectilinear
    from pySDC.helpers.vtkIO import readFromVTR

    coords = [np.linspace(0, 1, num=n, endpoint=False) for n in [nX, nY, nZ]]
    file = Rectilinear(np.float64, "testToVTR.pysdc")
    file.setHeader(nVar=nVar, coords=coords)
    file.initialize()
    u0 = np.random.rand(nVar, nX, nY, nZ).astype(file.dtype)
    times = np.arange(nSteps) / nSteps
    for t in times:
        ut = (u0 * t).astype(file.dtype)
        file.addField(t, ut)

    # Cleaning after eventuall other tests ...
    for f in glob.glob("testToVTR*.vtr"):
        os.remove(f)

    file.toVTR("testToVTR", varNames=[f"var{i}" for i in range(nVar)])

    vtrFiles = glob.glob("testToVTR*.vtr")
    assert len(vtrFiles) == file.nFields

    vtrFiles.sort(key=lambda name: int(name.split("_")[-1][:-4]))
    for i, vFile in enumerate(vtrFiles):
        uVTR, coords, _ = readFromVTR(vFile)
        _, uFile = file.readField(i)
        assert np.allclose(uFile, uVTR), "mismatch between data"
    for i, (xVTR, xFile) in enumerate(zip(coords, file.header["coords"])):
        assert np.allclose(xVTR, xFile), f"coordinate mismatch in dir. {i}"


@pytest.mark.mpi4py
@pytest.mark.parametrize("nVar", [1, 4])
@pytest.mark.parametrize("nSteps", [1, 10])
@pytest.mark.parametrize("algo", ["ChatGPT", "Hybrid"])
@pytest.mark.parametrize("dtypeIdx", [0, 1])
@pytest.mark.parametrize("nProcs", [2, 4])
@pytest.mark.parametrize("dim", [2, 3])
def testRectilinear_MPI(dim, nProcs, dtypeIdx, algo, nSteps, nVar):

    import subprocess

    fileName = f"testRectilinear{dim}D_MPI.pysdc"

    for gridSizes in itertools.product(*[[61, 16]] * dim):

        cmd = f"mpirun -np {nProcs} python {__file__} --fileName {fileName}"
        cmd += f" --dtypeIdx {dtypeIdx} --algo {algo} --nSteps {nSteps} --nVar {nVar} --gridSizes {' '.join([str(n) for n in gridSizes])}"

        p = subprocess.Popen(cmd.split(), cwd=".")
        p.wait()
        assert p.returncode == 0, f"MPI write with {nProcs} proc(s) did not return code 0, but {p.returncode}"

        from pySDC.helpers.fieldsIO import Rectilinear, initGrid

        f2: Rectilinear = FieldsIO.fromFile(fileName)

        assert type(f2) == Rectilinear, f"incorrect type in MPI written fields {f2}"
        assert f2.nFields == nSteps, f"incorrect nFields in MPI written fields {f2} ({f2.nFields} instead of {nSteps})"
        assert f2.nVar == nVar, f"incorrect nVar in MPI written fields {f2}"
        assert f2.gridSizes == list(gridSizes), f"incorrect gridSizes in MPI written fields {f2}"

        coords, u0 = initGrid(nVar, gridSizes)
        for i, (cFile, cRef) in enumerate(zip(f2.header['coords'], coords)):
            assert np.allclose(cFile, cRef), f"incorrect coords[{i}] in MPI written fields {f2}"

        times = np.arange(nSteps) / nSteps
        for idx, t in enumerate(times):
            u1 = u0 * t
            t2, u2 = f2.readField(idx)
            assert t2 == t, f"fields[{idx}] in {f2} has incorrect time ({t2} instead of {t})"
            assert u2.shape == u1.shape, f"{idx}'s fields in {f2} has incorrect shape"
            assert np.allclose(u2, u1), f"{idx}'s fields in {f2} has incorrect values"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--fileName', type=str, help='fileName of the file')
    parser.add_argument('--dtypeIdx', type=int, help="dtype index", choices=DTYPES.keys())
    parser.add_argument('--algo', type=str, help="algorithm used for block decomposition")
    parser.add_argument('--nSteps', type=int, help="number of time-steps")
    parser.add_argument('--nVar', type=int, help="number of field variables")
    parser.add_argument('--gridSizes', type=int, nargs='+', help="number of grid points in each dimensions")
    args = parser.parse_args()

    if sys.version_info >= (3, 11):
        from pySDC.helpers.fieldsIO import writeFields_MPI, compareFields_MPI

        u0 = writeFields_MPI(**args.__dict__)
        compareFields_MPI(args.fileName, u0, args.nSteps)
