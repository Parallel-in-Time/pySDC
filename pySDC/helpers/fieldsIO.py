#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic utility class to write and read cartesian grid field solutions into binary files.
It implements the base file handler class :class:`FieldsIO`, that is specialized into :

- :class:`Scal0D` : for 0D fields (scalar) with a given number of variables
- :class:`Cart1D` : for 1D fields with a given number of variables
- :class:`Cart2D` : for 2D fields on a cartesian grid with a given number of variables

While each file handler need to be setup with specific parameters (grid, ...),
each written file can be read using the same interface implemented in the
base abstract class.

Example
-------
>>> import numpy as np
>>> from pySDC.helpers.fieldsIO import Cart1D, FieldsIO
>>>
>>> # Write some fields in files
>>> x = np.linspace(0, 1, 101)
>>> fOut = Cart2D(np.float64, "file.pysdc")
>>> fOut.setHeader(nVar=2, coordX=x)
>>> fOut.initialize()
>>> times = [0, 1, 2]
>>> u0 = np.array([-1, 1])[:, None]*x[None, :]
>>> for t in times:
>>>    fOut.addField(t, t*u0)
>>>
>>> # Read the file using a the generic interface
>>> fIn = FieldsIO.fromFile("file.pysdc")
>>> times = fIn.times
>>> assert len(times) == fIn.nFields
>>> tEnd, uEnd = fIn.readField(-1)
>>> assert tEnd == times[-1]

Notes
-----
🚀 :class:`Cart1D` and :class:`Cart2D` are compatible with a MPI-based cartesian decomposition.
See :class:`pySDC.tests.test_helpers.test_fieldsIO.writeFields_MPI` for an illustrative example.

Warning
-------
To use MPI collective writing, you need to call first the class methods :class:`Cart1D.initMPI`
or :class:`Cart2D.initMPI` from the associated class (cf their docstring).
Also, their associated `setHeader` methods **must be given the global grids coordinates**,
wether code is run in parallel or not.
"""
import os
import numpy as np
from typing import Type, TypeVar

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
    2: np.float128,  # quadruple precision
    3: np.complex256,
    4: np.float32,  # single precision
    5: np.complex64,
}
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
        assert dtype in DTYPES_AVAIL, f"{dtype=} not available"
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
class Scal0D(FieldsIO):
    """FieldsIO handler storing a given number of scalar"""

    # -------------------------------------------------------------------------
    # Overriden methods
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
class Cart1D(Scal0D):
    """FieldsIO handler storing a given number of 2D scalar variables"""

    # -------------------------------------------------------------------------
    # Overriden methods
    # -------------------------------------------------------------------------
    def setHeader(self, nVar, coordX):
        """
        Set the descriptive grid structure to be stored in the file header.

        Parameters
        ----------
        nVar : int
            Number of 1D variables stored.
        coordX : np.1darray
            The grid coordinates in X direction.

        Note
        ----
        When used in MPI decomposition mode, `coordX` **must** be the global grid.
        """
        coords = self.setupCoords(coordX=coordX)
        self.header = {"nVar": int(nVar), **coords}
        self.nItems = nVar * self.nX

    @property
    def hInfos(self):
        """Array representing the grid structure to be written in the binary file."""
        return [np.array([self.nVar, self.nX], dtype=np.int64), np.array(self.header["coordX"], dtype=np.float64)]

    def readHeader(self, f):
        """
        Read the header from the binary file storing the fields.

        Parameters
        ----------
        f : `_io.TextIOWrapper`
            File to read the header from.
        """
        nVar, nX = np.fromfile(f, dtype=np.int64, count=2)
        coordX = np.fromfile(f, dtype=np.float64, count=nX)
        self.setHeader(nVar, coordX)

    def reshape(self, fields: np.ndarray):
        fields.shape = (self.nVar, self.nX)

    # -------------------------------------------------------------------------
    # Class specifics
    # -------------------------------------------------------------------------
    @property
    def nX(self):
        """Number of points in x direction"""
        return self.header["coordX"].size

    @staticmethod
    def setupCoords(**coords):
        """Utility function to setup grids in multuple dimensions, given the keyword arguments"""
        coords = {name: np.asarray(coord, dtype=np.float64) for name, coord in coords.items()}
        for name, coord in coords.items():
            assert coord.ndim == 1, f"{name} must be one dimensional"
        return coords

    # -------------------------------------------------------------------------
    # MPI-parallel implementation
    # -------------------------------------------------------------------------
    comm: MPI.Intracomm = None

    @classmethod
    def setupMPI(cls, comm: MPI.Intracomm, iLocX, nLocX):
        """
        Setup the MPI mode for the files IO, considering a decomposition
        of the 1D grid into contiuous subintervals.

        Parameters
        ----------
        comm : MPI.Intracomm
            The space decomposition communicator.
        iLocX : int
            Starting index of the local sub-domain in the global `coordX`.
        nLocX : int
            Number of points in the local sub-domain.
        """
        cls.comm = comm
        cls.iLocX = iLocX
        cls.nLocX = nLocX
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
        assert field.shape == (self.nVar, self.nLocX), f"expected {(self.nVar, self.nLocX)} shape, got {field.shape}"

        offset0 = self.fileSize
        self.MPI_FILE_OPEN(mode="a")
        if self.MPI_ROOT:
            self.MPI_WRITE(np.array(time, dtype=T_DTYPE))
        offset0 += self.tSize

        for iVar in range(self.nVar):
            offset = offset0 + (iVar * self.nX + self.iLocX) * self.itemSize
            self.MPI_WRITE_AT(offset, field[iVar])
        self.MPI_FILE_CLOSE()

    def readField(self, idx):
        """
        Read one field stored in the binary file, corresponding to the given
        time index, possibly in MPI mode.

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

        offset0 = self.hSize + idx * (self.fSize + self.tSize)
        with open(self.fileName, "rb") as f:
            t = float(np.fromfile(f, dtype=T_DTYPE, count=1, offset=offset0)[0])
        offset0 += self.tSize

        field = np.empty((self.nVar, self.nLocX), dtype=self.dtype)

        self.MPI_FILE_OPEN(mode="r")
        for iVar in range(self.nVar):
            offset = offset0 + (iVar * self.nX + self.iLocX) * self.itemSize
            self.MPI_READ_AT(offset, field[iVar])
        self.MPI_FILE_CLOSE()

        return t, field


@FieldsIO.register(sID=2)
class Cart2D(Cart1D):
    """FieldsIO handler storing a given number of 2D scalar variables"""

    # -------------------------------------------------------------------------
    # Overriden methods
    # -------------------------------------------------------------------------
    def setHeader(self, nVar, coordX, coordY):
        """
        Set the descriptive grid structure to be stored in the file header.

        Parameters
        ----------
        nVar : int
            Number of 1D variables stored.
        coordX : np.1darray
            The grid coordinates in x direction.
        coordY : np.1darray
            The grid coordinates in y direction.

        Note
        ----
        When used in MPI decomposition mode, `coordX` and `coordX` **must** be the global grid.
        """
        coords = self.setupCoords(coordX=coordX, coordY=coordY)
        self.header = {"nVar": int(nVar), **coords}
        self.nItems = nVar * self.nX * self.nY

    @property
    def hInfos(self):
        """Array representing the grid structure to be written in the binary file."""
        return [
            np.array([self.nVar, self.nX, self.nY], dtype=np.int64),
            np.array(self.header["coordX"], dtype=np.float64),
            np.array(self.header["coordY"], dtype=np.float64),
        ]

    def readHeader(self, f):
        """
        Read the header from the binary file storing the fields.

        Parameters
        ----------
        f : `_io.TextIOWrapper`
            File to read the header from.
        """
        nVar, nX, nY = np.fromfile(f, dtype=np.int64, count=3)
        coordX = np.fromfile(f, dtype=np.float64, count=nX)
        coordY = np.fromfile(f, dtype=np.float64, count=nY)
        self.setHeader(nVar, coordX, coordY)

    def reshape(self, fields: np.ndarray):
        """Reshape the fields to a [nVar, nX, nY] array (inplace operation)"""
        fields.shape = (self.nVar, self.nX, self.nY)

    # -------------------------------------------------------------------------
    # Class specifics
    # -------------------------------------------------------------------------
    @property
    def nY(self):
        """Number of points in y direction"""
        return self.header["coordY"].size

    # -------------------------------------------------------------------------
    # MPI-parallel implementation
    # -------------------------------------------------------------------------
    @classmethod
    def setupMPI(cls, comm: MPI.Intracomm, iLocX, nLocX, iLocY, nLocY):
        super().setupMPI(comm, iLocX, nLocX)
        cls.iLocY = iLocY
        cls.nLocY = nLocY

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
            self.nLocX,
            self.nLocY,
        ), f"expected {(self.nVar, self.nLocX, self.nLocY)} shape, got {field.shape}"

        offset0 = self.fileSize
        self.MPI_FILE_OPEN(mode="a")
        if self.MPI_ROOT:
            self.MPI_WRITE(np.array(time, dtype=T_DTYPE))
        offset0 += self.tSize

        for iVar in range(self.nVar):
            for iX in range(self.nLocX):
                offset = offset0 + (iVar * self.nX * self.nY + (self.iLocX + iX) * self.nY + self.iLocY) * self.itemSize
                self.MPI_WRITE_AT(offset, field[iVar, iX])
        self.MPI_FILE_CLOSE()

    def readField(self, idx):
        """
        Read one field stored in the binary file, corresponding to the given
        time index, eventually in MPI mode.

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

        field = np.empty((self.nVar, self.nLocX, self.nLocY), dtype=self.dtype)

        self.MPI_FILE_OPEN(mode="r")
        for iVar in range(self.nVar):
            for iX in range(self.nLocX):
                offset = offset0 + (iVar * self.nX * self.nY + (self.iLocX + iX) * self.nY + self.iLocY) * self.itemSize
                self.MPI_READ_AT(offset, field[iVar, iX])
        self.MPI_FILE_CLOSE()

        return t, field
