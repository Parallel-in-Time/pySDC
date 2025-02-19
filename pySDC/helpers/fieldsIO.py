#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic utility class to write and read cartesian grid field solutions into binary files.
It implements the base file handler class :class:`FieldsIO`, that is specialized into :

- :class:`Scalar` : for 0D fields (scalar) with a given number of variables
- :class:`Rectilinear` : for fields on N-dimensional rectilinear grids

While each file handler need to be setup with specific parameters (grid, ...),
each written file can be read using the same interface implemented in the
base abstract class.

Example
-------
>>> import numpy as np
>>> from pySDC.helpers.fieldsIO import Rectilinear
>>>
>>> # Write some fields in files
>>> x = np.linspace(0, 1, 128)
>>> y = np.linspace(0, 1, 64)
>>> fOut = Rectilinear(np.float64, "file.pysdc")
>>> fOut.setHeader(nVar=2, coords=[x, y])
>>> fOut.initialize()
>>> times = [0, 1, 2]
>>> xGrid, yGrid = np.meshgrid(x, y, indexing="ij")
>>> u0 = np.array([-1, 1]).reshape((-1, 1, 1))*xGrid*yGrid
>>> # u0 has shape [2, nX, nY]
>>> for t in times:
>>>    fOut.addField(t, t*u0)
>>>
>>> # Read the file using the generic interface
>>> from pySDC.helpers.fieldsIO import FieldsIO
>>> fIn = FieldsIO.fromFile("file.pysdc")
>>> times = fIn.times
>>> assert len(times) == fIn.nFields
>>> tEnd, uEnd = fIn.readField(-1)
>>> assert tEnd == times[-1]

Notes
-----
🚀 :class:`Rectilinear` is compatible with a MPI-based cartesian decomposition.
See :class:`pySDC.helpers.fieldsIO.writeFields_MPI` for an illustrative example.

Warning
-------
To use MPI collective writing, you need to call first the class methods :class:`Rectilinear.initMPI` (cf their docstring).
Also, `Rectilinear.setHeader` **must be given the global grids coordinates**, wether the code is run in parallel or not.

> ⚠️ Also : this module can only be imported with **Python 3.11 or higher** !
"""
import os
import numpy as np
from typing import Type, TypeVar
import logging
import itertools

T = TypeVar("T")

try:
    try:
        import dolfin as df  # noqa: F841 (for some reason, dolfin always needs to be imported before mpi4py)
    except ImportError:
        pass
    from mpi4py import MPI
except ImportError:

    class MPI:
        COMM_WORLD = None
        Intracomm = T


# Supported data types
DTYPES = {
    0: np.float64,  # double precision
    1: np.complex128,
}
try:
    DTYPES.update(
        {
            2: np.float128,  # quadruple precision
            3: np.complex256,
        }
    )
except AttributeError:
    logging.getLogger('FieldsIO').debug('Warning: Quadruple precision not available on this machine')
try:
    DTYPES.update(
        {
            4: np.float32,  # single precision
            5: np.complex64,
        }
    )
except AttributeError:
    logging.getLogger('FieldsIO').debug('Warning: Single precision not available on this machine')

DTYPES_AVAIL = {val: key for key, val in DTYPES.items()}

# Header dtype
H_DTYPE = np.int8
T_DTYPE = np.float64


class FieldsIO:
    """Abstract IO file handler"""

    STRUCTS = {}  # supported structures, modified dynamically
    sID = None  # structure ID of the FieldsIO class, modified dynamically

    tSize = T_DTYPE().itemsize

    ALLOW_OVERWRITE = False

    def __init__(self, dtype, fileName):
        """
        Parameters
        ----------
        dtype : np.dtype
            The data type of the fields values.
        fileName : str
            File.
        """
        assert dtype in DTYPES_AVAIL, f"{dtype=} not available. Supported on this machine: {list(DTYPES_AVAIL.keys())}"
        self.dtype = dtype
        self.fileName = fileName
        self.initialized = False

        # Initialized by the setHeader abstract method
        self.header = None
        self.nItems = None  # number of values (dof) stored into one field

    def __str__(self):
        return f"FieldsIO[{self.__class__.__name__}|{self.dtype.__name__}|file:{self.fileName}]<{hex(id(self))}>"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def fromFile(cls, fileName):
        """
        Read a file storing fields, and return the `FieldsIO` of the appropriate
        field type (structure).

        Parameters
        ----------
        fileName : str
            Name of the binary file.

        Returns
        -------
        fieldsIO : :class:`FieldsIO`
            The specialized `FieldsIO` adapted to the file.
        """
        assert os.path.isfile(fileName), f"not a file ({fileName})"
        with open(fileName, "rb") as f:
            STRUCT, DTYPE = np.fromfile(f, dtype=H_DTYPE, count=2)
            fieldsIO: FieldsIO = cls.STRUCTS[STRUCT](DTYPES[DTYPE], fileName)
            fieldsIO.readHeader(f)
            fieldsIO.initialized = True
        return fieldsIO

    @property
    def hBase(self) -> np.ndarray:
        """Base header into numpy array format"""
        return np.array([self.sID, DTYPES_AVAIL[self.dtype]], dtype=H_DTYPE)

    @classmethod
    def register(cls, sID):
        """
        Decorator used to register a new class FieldsIO specialized class

        Parameters
        ----------
        sID : int
            Unique identifyer for the file, used in the binary file.
            Since it's encoded on a 8-bytes signed integer,
            it must be between -128 and 127

        Example
        -------
        >>> # New specialized FieldsIO class
        >>> @FieldsIO.register(sID=31)
        >>> class HexaMesh2D(FieldsIO):
        >>>     pass # ... implementation
        """

        def wrapper(registered: Type[T]) -> Type[T]:
            assert (
                sID not in cls.STRUCTS
            ), f"struct ID already taken by {cls.STRUCTS[sID]}, cannot use it for {registered}"
            cls.STRUCTS[sID] = registered
            registered.sID = sID
            return registered

        return wrapper

    def initialize(self):
        """Initialize the file handler : create the file with header, removing any existing file with the same name"""
        assert self.header is not None, "header must be set before initializing FieldsIO"
        assert not self.initialized, "FieldsIO already initialized"

        if not self.ALLOW_OVERWRITE:
            assert not os.path.isfile(
                self.fileName
            ), "file already exists, use FieldsIO.ALLOW_OVERWRITE = True to allow overwriting"

        with open(self.fileName, "w+b") as f:
            self.hBase.tofile(f)
            for array in self.hInfos:
                array.tofile(f)
        self.initialized = True

    def setHeader(self, **params):
        """(Abstract) Set the header before creating a new file to store the fields"""
        raise NotImplementedError()

    @property
    def hInfos(self) -> list[np.ndarray]:
        """(Abstract) Array representing the grid structure to be written in the binary file."""
        raise NotImplementedError()

    def readHeader(self, f):
        """
        (Abstract) Read the header from the file storing the fields.

        Parameters
        ----------
        f : `_io.TextIOWrapper`
            File to read the header from.
        """
        raise NotImplementedError()

    @property
    def hSize(self):
        """Size of the full header (in bytes)"""
        return self.hBase.nbytes + sum(hInfo.nbytes for hInfo in self.hInfos)

    @property
    def itemSize(self):
        """Size of one field value (in bytes)"""
        return self.dtype().itemsize

    @property
    def fSize(self):
        """Full size of a field (in bytes)"""
        return self.nItems * self.itemSize

    @property
    def fileSize(self):
        """Current size of the file (in bytes)"""
        return os.path.getsize(self.fileName)

    def addField(self, time, field):
        """
        Append one field solution at the end of the file with one given time.

        Parameters
        ----------
        time : float-like
            The associated time of the field solution.
        field : np.ndarray
            The field values.
        """
        assert self.initialized, "cannot add field to a non initialized FieldsIO"
        field = np.asarray(field)
        assert field.dtype == self.dtype, f"expected {self.dtype} dtype, got {field.dtype}"
        assert field.size == self.nItems, f"expected {self.nItems} values, got {field.size}"
        with open(self.fileName, "ab") as f:
            np.array(time, dtype=T_DTYPE).tofile(f)
            field.tofile(f)

    @property
    def nFields(self):
        """Number of fields currently stored in the binary file"""
        return int((self.fileSize - self.hSize) // (self.tSize + self.fSize))

    def formatIndex(self, idx):
        """Utility method to format a fields index to a positional integer (negative starts from last field index, like python lists)"""
        nFields = self.nFields
        if idx < 0:
            idx = nFields + idx
        assert idx < nFields, f"cannot read index {idx} from {nFields} fields"
        assert idx >= 0, f"cannot read index {idx-nFields} from {nFields} fields"
        return idx

    @property
    def times(self):
        """Vector of all times stored in the binary file"""
        times = []
        with open(self.fileName, "rb") as f:
            f.seek(self.hSize)
            for i in range(self.nFields):
                t = np.fromfile(f, dtype=T_DTYPE, count=1, offset=0 if i == 0 else self.fSize)[0]
                times.append(float(t))
        return times

    def time(self, idx):
        """Time stored at a given field index"""
        idx = self.formatIndex(idx)
        offset = self.hSize + idx * (self.tSize + self.fSize)
        with open(self.fileName, "rb") as f:
            t = np.fromfile(f, dtype=T_DTYPE, count=1, offset=offset)[0]
        return float(t)

    def readField(self, idx):
        """
        Read one field stored in the binary file, corresponding to the given
        time index.

        Parameters
        ----------
        idx : int
            Positional index of the field.

        Returns
        -------
        t : float
            Stored time for this field.
        field : np.ndarray
            Read fields in a numpy array.
        """
        idx = self.formatIndex(idx)
        offset = self.hSize + idx * (self.tSize + self.fSize)
        with open(self.fileName, "rb") as f:
            f.seek(offset)
            t = float(np.fromfile(f, dtype=T_DTYPE, count=1)[0])
            field = np.fromfile(f, dtype=self.dtype, count=self.nItems)
        self.reshape(field)
        return t, field

    def reshape(self, field):
        """Eventually reshape the field to correspond to the grid structure"""
        pass


@FieldsIO.register(sID=0)
class Scalar(FieldsIO):
    """FieldsIO handler storing a given number of scalar"""

    # -------------------------------------------------------------------------
    # Overridden methods
    # -------------------------------------------------------------------------
    def setHeader(self, nVar):
        """
        Set the descriptive grid structure to be stored in the file header.

        Parameters
        ----------
        nVar : int
            Number of scalar variable stored.
        """
        self.header = {"nVar": int(nVar)}
        self.nItems = self.nVar

    @property
    def hInfos(self):
        """Array representing the grid structure to be written in the binary file."""
        return [np.array([self.nVar], dtype=np.int64)]

    def readHeader(self, f):
        """
        Read the header from the binary file storing the fields.

        Parameters
        ----------
        f : `_io.TextIOWrapper`
            File to read the header from.
        """
        (nVar,) = np.fromfile(f, dtype=np.int64, count=1)
        self.setHeader(nVar)

    # -------------------------------------------------------------------------
    # Class specifics
    # -------------------------------------------------------------------------
    @property
    def nVar(self):
        """Number of variables in a fields, as described in the header"""
        return self.header["nVar"]


@FieldsIO.register(sID=1)
class Rectilinear(Scalar):
    """FieldsIO handler storing a given number of scalar variables on a N-dimensional rectilinear grid"""

    @staticmethod
    def setupCoords(*coords):
        """Utility function to setup grids in multiple dimensions, given the keyword arguments"""
        coords = [np.asarray(coord, dtype=np.float64) for coord in coords]
        for axis, coord in enumerate(coords):
            assert coord.ndim == 1, f"coord for {axis=} must be one dimensional"
        return coords

    # -------------------------------------------------------------------------
    # Overridden methods
    # -------------------------------------------------------------------------
    def setHeader(self, nVar, coords):
        """
        Set the descriptive grid structure to be stored in the file header.

        Parameters
        ----------
        nVar : int
            Number of 1D variables stored.
        coords : np.1darray or list[np.1darray]
            The grid coordinates in each dimensions.

        Note
        ----
        When used in MPI decomposition mode, all coordinate **must** be the global grid.
        """
        if not isinstance(coords, (tuple, list)):
            coords = [coords]
        coords = self.setupCoords(*coords)
        self.header = {"nVar": int(nVar), "coords": coords}
        self.nItems = nVar * self.nDoF

    @property
    def hInfos(self):
        """Array representing the grid structure to be written in the binary file."""
        return [np.array([self.nVar, self.dim, *self.gridSizes], dtype=np.int32)] + [
            np.array(coord, dtype=np.float64) for coord in self.header["coords"]
        ]

    def readHeader(self, f):
        """
        Read the header from the binary file storing the fields.

        Parameters
        ----------
        f : `_io.TextIOWrapper`
            File to read the header from.
        """
        nVar, dim = np.fromfile(f, dtype=np.int32, count=2)
        gridSizes = np.fromfile(f, dtype=np.int32, count=dim)
        coords = [np.fromfile(f, dtype=np.float64, count=n) for n in gridSizes]
        self.setHeader(nVar, coords)

    def reshape(self, fields: np.ndarray):
        """Reshape the fields to a N-d array (inplace operation)"""
        fields.shape = (self.nVar, *self.gridSizes)

    # -------------------------------------------------------------------------
    # Class specifics
    # -------------------------------------------------------------------------
    @property
    def gridSizes(self):
        """Number of points in y direction"""
        return [coord.size for coord in self.header["coords"]]

    @property
    def dim(self):
        """Number of grid dimensions"""
        return len(self.gridSizes)

    @property
    def nDoF(self):
        """Number of degrees of freedom for one variable"""
        return np.prod(self.gridSizes)

    def toVTR(self, baseName, varNames, idxFormat="{:06d}"):
        """
        Convert all 3D fields stored in binary format (FieldsIO) into a list
        of VTR files, that can be read later with Paraview or equivalent to
        make videos.

        Parameters
        ----------
        baseName : str
            Base name of the VTR file.
        varNames : list[str]
            Variable names of the fields.
        idxFormat : str, optional
            Formatting string for the index of the VTR file.
            The default is "{:06d}".

        Example
        -------
        >>> # Suppose the FieldsIO object is already writen into outputs.pysdc
        >>> import os
        >>> from pySDC.utils.fieldsIO import Rectilinear
        >>> os.makedirs("vtrFiles")  # to store all VTR files into a subfolder
        >>> Rectilinear.fromFile("outputs.pysdc").toVTR(
        >>>    baseName="vtrFiles/field", varNames=["u", "v", "w", "T", "p"])
        """
        assert self.dim == 3, "can only be used with 3D fields"
        from pySDC.helpers.vtkIO import writeToVTR

        template = f"{baseName}_{idxFormat}"
        for i in range(self.nFields):
            _, u = self.readField(i)
            writeToVTR(template.format(i), u, self.header["coords"], varNames)

    # -------------------------------------------------------------------------
    # MPI-parallel implementation
    # -------------------------------------------------------------------------
    comm: MPI.Intracomm = None

    @classmethod
    def setupMPI(cls, comm: MPI.Intracomm, iLoc, nLoc):
        """
        Setup the MPI mode for the files IO, considering a decomposition
        of the 1D grid into contiuous subintervals.

        Parameters
        ----------
        comm : MPI.Intracomm
            The space decomposition communicator.
        iLoc : list[int]
            Starting index of the local sub-domain in the global coordinates.
        nLoc : list[int]
            Number of points in the local sub-domain.
        """
        cls.comm = comm
        cls.iLoc = iLoc
        cls.nLoc = nLoc
        cls.mpiFile = None

    @property
    def MPI_ON(self):
        """Wether or not MPI is activated"""
        if self.comm is None:
            return False
        return self.comm.Get_size() > 1

    @property
    def MPI_ROOT(self):
        """Wether or not the process is MPI Root"""
        if self.comm is None:
            return True
        return self.comm.Get_rank() == 0

    def MPI_FILE_OPEN(self, mode):
        """Open the binary file in MPI mode"""
        amode = {
            "r": MPI.MODE_RDONLY,
            "a": MPI.MODE_WRONLY | MPI.MODE_APPEND,
        }[mode]
        self.mpiFile = MPI.File.Open(self.comm, self.fileName, amode)

    def MPI_WRITE(self, data):
        """Write data (np.ndarray) in the binary file in MPI mode, at the current file cursor position."""
        self.mpiFile.Write(data)

    def MPI_WRITE_AT(self, offset, data: np.ndarray):
        """
        Write data in the binary file in MPI mode, with a given offset
        **relative to the beginning of the file**.

        Parameters
        ----------
        offset : int
            Offset to write at, relative to the beginning of the file, in bytes.
        data : np.ndarray
            Data to be written in the binary file.
        """
        self.mpiFile.Write_at(offset, data)

    def MPI_READ_AT(self, offset, data):
        """
        Read data from the binary file in MPI mode, with a given offset
        **relative to the beginning of the file**.

        Parameters
        ----------
        offset : int
            Offset to read at, relative to the beginning of the file, in bytes.
        data : np.ndarray
            Array on which to read the data from the binary file.
        """
        self.mpiFile.Read_at(offset, data)

    def MPI_FILE_CLOSE(self):
        """Close the binary file in MPI mode"""
        self.mpiFile.Close()
        self.mpiFile = None

    def initialize(self):
        """Initialize the binary file (write header) in MPI mode"""
        if self.MPI_ROOT:
            try:
                super().initialize()
            except AssertionError as e:
                if self.MPI_ON:
                    print(f"{type(e)}: {e}")
                    self.comm.Abort()
                else:
                    raise e

        if self.MPI_ON:
            self.comm.Barrier()  # Important, should not be removed !
            self.initialized = True

    def addField(self, time, field):
        """
        Append one field solution at the end of the file with one given time,
        possibly using MPI.

        Parameters
        ----------
        time : float-like
            The associated time of the field solution.
        field : np.ndarray
            The (local) field values.

        Note
        ----
        If a MPI decomposition is used, field **must be** the local field values.
        """
        if not self.MPI_ON:
            return super().addField(time, field)

        assert self.initialized, "cannot add field to a non initialized FieldsIO"

        field = np.asarray(field)
        assert field.dtype == self.dtype, f"expected {self.dtype} dtype, got {field.dtype}"
        assert field.shape == (
            self.nVar,
            *self.nLoc,
        ), f"expected {(self.nVar, *self.nLoc)} shape, got {field.shape}"

        offset0 = self.fileSize
        self.MPI_FILE_OPEN(mode="a")
        if self.MPI_ROOT:
            self.MPI_WRITE(np.array(time, dtype=T_DTYPE))
        offset0 += self.tSize

        for (iVar, *iBeg) in itertools.product(range(self.nVar), *[range(n) for n in self.nLoc[:-1]]):
            offset = offset0 + self.iPos(iVar, iBeg) * self.itemSize
            self.MPI_WRITE_AT(offset, field[iVar, *iBeg])
        self.MPI_FILE_CLOSE()

    def iPos(self, iVar, iX):
        iPos = iVar * self.nDoF
        for axis in range(self.dim - 1):
            iPos += (self.iLoc[axis] + iX[axis]) * np.prod(self.gridSizes[axis + 1 :])
        iPos += self.iLoc[-1]
        return iPos

    def readField(self, idx):
        """
        Read one field stored in the binary file, corresponding to the given
        time index, using MPI in the eventuality of space parallel decomposition.

        Parameters
        ----------
        idx : int
            Positional index of the field.

        Returns
        -------
        t : float
            Stored time for this field.
        field : np.ndarray
            Read (local) fields in a numpy array.

        Note
        ----
        If a MPI decomposition is used, it reads and returns the local fields values only.
        """
        if not self.MPI_ON:
            return super().readField(idx)

        idx = self.formatIndex(idx)
        offset0 = self.hSize + idx * (self.tSize + self.fSize)
        with open(self.fileName, "rb") as f:
            t = float(np.fromfile(f, dtype=T_DTYPE, count=1, offset=offset0)[0])
        offset0 += self.tSize

        field = np.empty((self.nVar, *self.nLoc), dtype=self.dtype)

        self.MPI_FILE_OPEN(mode="r")
        for (iVar, *iBeg) in itertools.product(range(self.nVar), *[range(n) for n in self.nLoc[:-1]]):
            offset = offset0 + self.iPos(iVar, iBeg) * self.itemSize
            self.MPI_READ_AT(offset, field[iVar, *iBeg])
        self.MPI_FILE_CLOSE()

        return t, field


# -----------------------------------------------------------------------------------------------
# Utility functions used for testing
# -----------------------------------------------------------------------------------------------
def initGrid(nVar, gridSizes):
    dim = len(gridSizes)
    coords = [np.linspace(0, 1, num=n, endpoint=False) for n in gridSizes]
    s = [None] * dim
    u0 = np.array(np.arange(nVar) + 1)[:, *s]
    for x in np.meshgrid(*coords, indexing="ij"):
        u0 = u0 * x
    return coords, u0


def writeFields_MPI(fileName, dtypeIdx, algo, nSteps, nVar, gridSizes):
    coords, u0 = initGrid(nVar, gridSizes)

    from mpi4py import MPI
    from pySDC.helpers.blocks import BlockDecomposition
    from pySDC.helpers.fieldsIO import Rectilinear

    comm = MPI.COMM_WORLD
    MPI_SIZE = comm.Get_size()
    MPI_RANK = comm.Get_rank()

    blocks = BlockDecomposition(MPI_SIZE, gridSizes, algo, MPI_RANK)

    iLoc, nLoc = blocks.localBounds
    Rectilinear.setupMPI(comm, iLoc, nLoc)
    s = [slice(i, i + n) for i, n in zip(iLoc, nLoc)]
    u0 = u0[:, *s]
    print(MPI_RANK, u0.shape)

    f1 = Rectilinear(DTYPES[dtypeIdx], fileName)
    f1.setHeader(nVar=nVar, coords=coords)

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
        assert t2 == t, f"fields[{idx}] in {f2} has incorrect time ({t2} instead of {t})"
        assert u2.shape == u1.shape, f"{idx}'s fields in {f2} has incorrect shape"
        assert np.allclose(u2, u1), f"{idx}'s fields in {f2} has incorrect values"
