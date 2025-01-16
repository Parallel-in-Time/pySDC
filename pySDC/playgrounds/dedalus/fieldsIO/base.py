#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base generic script for fields IO
"""
import os
import sys
import numpy as np
from typing import Type, TypeVar
try:
    from mpi4py import MPI
except ImportError:
    pass


from time import time, sleep
from blocks import BlockDecomposition

T = TypeVar("T")

# Supported data types
DTYPES = {
    0: np.float64,      # double precision
    1: np.complex128,
    2: np.float128,     # quadruple precision
    3: np.complex256,
    4: np.float32,      # single precision
    5: np.complex64,
    }
DTYPES_AVAIL = {val: key for key, val in DTYPES.items()}

# Header dtype
H_DTYPE = np.int8
T_DTYPE = np.float64


class FieldsIO():

    STRUCTS = {}    # supported structures, modified dynamically
    sID = None      # structure ID of the FieldsIO class, modified dynamically

    tSize = T_DTYPE().itemsize

    def __init__(self, dtype, fileName):
        assert dtype in DTYPES_AVAIL , f"dtype not available ({dtype})"
        self.dtype = dtype
        self.fileName = fileName
        self.initialized = False

        # Initialized by the setHeader abstract method
        self.header = None
        self.nItems = None   # number of values (dof) stored into one field

    def __str__(self):
        return f"FieldsIO[{self.__class__.__name__}|{self.dtype.__name__}|file:{self.fileName}]<{hex(id(self))}>"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def fromFile(cls, fileName):
        assert os.path.isfile(fileName), f"not a file ({fileName})"
        with open(fileName, "rb") as f:
            STRUCT, DTYPE = np.fromfile(f, dtype=H_DTYPE, count=2)
            fieldsIO:FieldsIO = cls.STRUCTS[STRUCT](DTYPES[DTYPE], fileName)
            fieldsIO.readHeader(f)
            fieldsIO.initialized = True
        return fieldsIO

    @property
    def hBase(self)->np.ndarray:
        return np.array([self.sID, DTYPES_AVAIL[self.dtype]], dtype=H_DTYPE)

    @classmethod
    def register(cls, sID):
        def wrapper(registered:Type[T])->Type[T]:
            assert sID not in cls.STRUCTS, \
                f"struct ID already taken by {cls.STRUCTS[sID]}, cannot use it for {registered}"
            cls.STRUCTS[sID] = registered
            registered.sID = sID
            return registered
        return wrapper

    def initialize(self):
        assert self.header is not None, "header must be set before initializing FieldsIO"
        assert not self.initialized, "FieldsIO already initialized"

        with open(self.fileName, "w+b") as f:
            self.hBase.tofile(f)
            for array in self.hInfos:
                array.tofile(f)
        self.initialized = True

    def setHeader(self, **params):
        """Set the header before creating a new file to store the fields"""
        raise NotImplementedError()

    @property
    def hInfos(self)->list[np.ndarray]:
        raise NotImplementedError()

    @property
    def hSize(self):
        """Size of the full header (in bytes)"""
        return self.hBase.nbytes + sum(hInfo.nbytes for hInfo in self.hInfos)

    @property
    def itemSize(self):
        return self.dtype().itemsize

    @property
    def fSize(self):
        """Full size of a field (in bytes)"""
        return self.nItems * self.itemSize

    @property
    def fileSize(self):
        return os.path.getsize(self.fileName)

    def readHeader(self, f):
        """Read the header from the file storing the fields"""
        raise NotImplementedError()

    def addField(self, time, field):
        assert self.initialized, "cannot add field to a non initialized FieldsIO"
        field = np.asarray(field)
        assert field.dtype == self.dtype, f"expected {self.dtype} dtype, got {field.dtype}"
        assert field.size == self.nItems, f"expected {self.nItems} values, got {field.size}"
        with open(self.fileName, "ab") as f:
            np.array(time, dtype=T_DTYPE).tofile(f)
            field.tofile(f)

    @property
    def nFields(self):
        return int((self.fileSize-self.hSize)//(self.tSize + self.fSize))

    def check(self, idx):
        nFields = self.nFields
        if idx < 0:
            idx = nFields + idx
        assert idx < nFields, f"cannot read index {idx} from {nFields} fields"
        assert idx >= 0, f"cannot read index {idx-nFields} from {nFields} fields"
        return idx

    @property
    def times(self):
        times = []
        with open(self.fileName, "rb") as f:
            f.seek(self.hSize)
            for i in range(self.nFields):
                t = np.fromfile(f, dtype=T_DTYPE, count=1,
                                offset=0 if i == 0 else self.fSize)[0]
                times.append(float(t))
        return times

    def time(self, idx):
        idx = self.check(idx)
        offset = self.hSize + idx*(self.tSize + self.fSize)
        with open(self.fileName, "rb") as f:
            t = np.fromfile(f, dtype=T_DTYPE, count=1, offset=offset)[0]
        return float(t)


    def readField(self, idx):
        idx = self.check(idx)
        offset = self.hSize + idx*(self.tSize + self.fSize)
        with open(self.fileName, "rb") as f:
            f.seek(offset)
            t = float(np.fromfile(f, dtype=T_DTYPE, count=1)[0])
            field = np.fromfile(f, dtype=self.dtype, count=self.nItems)
        self.reshape(field)
        return t, field

    def reshape(self, field): pass


@FieldsIO.register(sID=0)
class Scal0D(FieldsIO):

    def setHeader(self, nVar):
        self.header = {"nVar": int(nVar)}
        self.nItems = self.nVar

    @property
    def nVar(self):
        return self.header["nVar"]

    @property
    def hInfos(self):
        return [
            np.array([self.nVar], dtype=np.int64)
        ]

    def readHeader(self, f):
        nVar, = np.fromfile(f, dtype=np.int64, count=1)
        self.setHeader(nVar)


@FieldsIO.register(sID=1)
class Cart1D(Scal0D):

    def setupGrids(self, **grids):
        grids = {name: np.asarray(grid, dtype=np.float64)
                 for name, grid in grids.items() if name.startswith("grid")}
        for name, grid in grids.items():
            assert grid.ndim == 1, f"{name} must be one dimensional"
        return grids

    def setHeader(self, nVar, gridX):
        grids = self.setupGrids(gridX=gridX)
        self.header = {"nVar": int(nVar), **grids}
        self.nItems = nVar*self.nX

    @property
    def nX(self):
        return self.header["gridX"].size

    def reshape(self, fields:np.ndarray):
        fields.shape = (self.nVar, self.nX)

    @property
    def hInfos(self):
        return [
            np.array([self.nVar, self.nX], dtype=np.int64),
            np.array(self.header["gridX"], dtype=np.float64)
        ]

    def readHeader(self, f):
        nVar, nX = np.fromfile(f, dtype=np.int64, count=2)
        gridX = np.fromfile(f, dtype=np.float64, count=nX)
        self.setHeader(nVar, gridX)

    # -------------------------------------------------------------------------
    # MPI-parallel implementation
    # -------------------------------------------------------------------------
    comm:MPI.Intracomm = None

    @classmethod
    def setupMPI(cls, comm:MPI.Intracomm, iLocX, nLocX):
        cls.comm = comm
        cls.iLocX = iLocX
        cls.nLocX = nLocX
        cls.mpiFile = None

    @property
    def MPI_ON(self):
        if self.comm is None: return False
        return self.comm.Get_size() > 1

    @property
    def MPI_ROOT(self):
        if self.comm is None: return True
        return self.comm.Get_rank() == 0

    def MPI_FILE_OPEN(self, mode):
        amode = {
            "r": MPI.MODE_RDONLY,
            "a": MPI.MODE_WRONLY | MPI.MODE_APPEND,
            }[mode]
        self.mpiFile = MPI.File.Open(self.comm, self.fileName, amode)

    def MPI_WRITE(self, data):
        self.mpiFile.Write(data)

    def MPI_WRITE_AT(self, offset, data:np.ndarray):
        self.mpiFile.Write_at(offset, data)

    def MPI_READ_AT(self, offset, data):
        self.mpiFile.Read_at(offset, data)

    def MPI_FILE_CLOSE(self):
        self.mpiFile.Close()
        self.mpiFile = None

    def initialize(self):
        if self.MPI_ROOT:
            super().initialize()
        if self.MPI_ON:
            self.comm.Barrier()
            self.initialized = True


    def addField(self, time, field):
        if not self.MPI_ON: return super().addField(time, field)

        assert self.initialized, "cannot add field to a non initialized FieldsIO"

        field = np.asarray(field)
        assert field.dtype == self.dtype, f"expected {self.dtype} dtype, got {field.dtype}"
        assert field.shape == (self.nVar, self.nLocX), \
            f"expected {(self.nVar, self.nLocX)} shape, got {field.shape}"

        offset0 = self.fileSize
        self.MPI_FILE_OPEN(mode="a")
        if self.MPI_ROOT:
            self.MPI_WRITE(np.array(time, dtype=T_DTYPE))
        offset0 += self.tSize

        for iVar in range(self.nVar):
            offset = offset0 + (iVar*self.nX + self.iLocX)*self.itemSize
            self.MPI_WRITE_AT(offset, field[iVar])
        self.MPI_FILE_CLOSE()


    def readField(self, idx):
        if not self.MPI_ON: return super().readField(idx)
        idx = self.check(idx)

        offset0 = self.hSize + idx*(self.fSize + self.tSize)
        with open(self.fileName, "rb") as f:
            t = float(np.fromfile(f, dtype=T_DTYPE, count=1, offset=offset0)[0])
        offset0 += self.tSize

        field = np.empty((self.nVar, self.nLocX), dtype=self.dtype)

        self.MPI_FILE_OPEN(mode="r")
        for iVar in range(self.nVar):
            offset = offset0 + (iVar*self.nX + self.iLocX)*self.itemSize
            self.MPI_READ_AT(offset, field[iVar])
        self.MPI_FILE_CLOSE()

        return t, field


@FieldsIO.register(sID=2)
class Cart2D(Cart1D):

    def setHeader(self, nVar, gridX, gridY):
        grids = self.setupGrids(gridX=gridX, gridY=gridY)
        self.header = {"nVar": int(nVar), **grids}
        self.nItems = nVar*self.nX*self.nY

    @property
    def nY(self):
        return self.header["gridY"].size

    def reshape(self, fields:np.ndarray):
        fields.shape = (self.nVar, self.nX, self.nY)

    @property
    def hInfos(self):
        return [
            np.array([self.nVar, self.nX, self.nY], dtype=np.int64),
            np.array(self.header["gridX"], dtype=np.float64),
            np.array(self.header["gridY"], dtype=np.float64),
        ]

    def readHeader(self, f):
        nVar, nX, nY = np.fromfile(f, dtype=np.int64, count=3)
        gridX = np.fromfile(f, dtype=np.float64, count=nX)
        gridY = np.fromfile(f, dtype=np.float64, count=nY)
        self.setHeader(nVar, gridX, gridY)


    # -------------------------------------------------------------------------
    # MPI-parallel implementation
    # -------------------------------------------------------------------------
    @classmethod
    def setupMPI(cls, comm:MPI.Intracomm, iLocX, nLocX, iLocY, nLocY):
        super().setupMPI(comm, iLocX, nLocX)
        cls.iLocY = iLocY
        cls.nLocY = nLocY


    def addField(self, time, field):
        if not self.MPI_ON: return super().addField(time, field)

        assert self.initialized, "cannot add field to a non initialized FieldsIO"

        field = np.asarray(field)
        assert field.dtype == self.dtype, f"expected {self.dtype} dtype, got {field.dtype}"
        assert field.shape == (self.nVar, self.nLocX, self.nLocY), \
            f"expected {(self.nVar, self.nLocX, self.nLocY)} shape, got {field.shape}"

        offset0 = self.fileSize
        self.MPI_FILE_OPEN(mode="a")
        if self.MPI_ROOT:
            self.MPI_WRITE(np.array(time, dtype=T_DTYPE))
        offset0 += self.tSize

        for iVar in range(self.nVar):
            for iX in range(self.nLocX):
                offset = offset0 + (
                    iVar*self.nX*self.nY + (self.iLocX + iX)*self.nY + self.iLocY
                    )*self.itemSize
                self.MPI_WRITE_AT(offset, field[iVar, iX])
        self.MPI_FILE_CLOSE()


    def readField(self, idx):
        if not self.MPI_ON: return super().readField(idx)
        idx = self.check(idx)

        offset0 = self.hSize + idx*(self.tSize + self.fSize)
        with open(self.fileName, "rb") as f:
            t = float(np.fromfile(f, dtype=T_DTYPE, count=1, offset=offset0)[0])
        offset0 += self.tSize

        field = np.empty((self.nVar, self.nLocX, self.nLocY), dtype=self.dtype)

        self.MPI_FILE_OPEN(mode="r")
        for iVar in range(self.nVar):
            for iX in range(self.nLocX):
                offset = offset0 + (
                    iVar*self.nX*self.nY + (self.iLocX + iX)*self.nY + self.iLocY
                    )*self.itemSize
                self.MPI_READ_AT(offset, field[iVar, iX])
        self.MPI_FILE_CLOSE()

        return t, field


if __name__ == "__main__":

    # f1 = Scal0D(np.float64, "test.pysdc")
    # f1.setHeader(nVar=10)
    # f1.initialize()

    # f1.addField(0.0, np.arange(10.0))
    # f1.addField(0.1, 2*np.arange(10.0))
    # f1.addField(0.2, 3*np.arange(10.0))
    # f1.addField(0.3, 4*np.arange(10.0))

    # f2 = FieldsIO.fromFile("test.pysdc")

    x = np.linspace(0, 1, num=256, endpoint=False)
    nX = x.size
    y = np.linspace(0, 1, num=64, endpoint=False)
    nY = y.size

    nDim = 1
    dType = np.float64

    if nDim == 1:
        u0 = np.array([-1, 1])[:, None]*x[None, :]
    if nDim == 2:
        u0 = np.array([-1, 1])[:, None, None]*x[None, :, None]*y[None, None, :]
    fileName = "test.pysdc"

    comm = MPI.COMM_WORLD
    MPI_SIZE = comm.Get_size()
    MPI_RANK = comm.Get_rank()

    gridSizes = u0.shape[1:]
    algo = sys.argv[1] if len(sys.argv) > 1 else "ChatGPT"
    blocks = BlockDecomposition(MPI_SIZE, gridSizes, algo, MPI_RANK)
    bounds = blocks.localBounds
    if MPI_SIZE > 1:
        fileName = "test_MPI.pysdc"


    if nDim == 1:
        (iLocX, ), (nLocX, ) = bounds
        pRankX, = blocks.ranks
        Cart1D.setupMPI(comm, iLocX, nLocX)
        u0 = u0[:, iLocX:iLocX+nLocX]

        MPI.COMM_WORLD.Barrier()
        sleep(0.01*MPI_RANK)
        print(f"[Rank {MPI_RANK}] pRankX={pRankX} ({iLocX}, {nLocX})")
        MPI.COMM_WORLD.Barrier()

        f1 = Cart1D(dType, fileName)
        f1.setHeader(nVar=u0.shape[0], gridX=x)

    if nDim == 2:
        (iLocX, iLocY), (nLocX, nLocY) = bounds
        pRankX, pRankY = blocks.ranks
        Cart2D.setupMPI(comm, iLocX, nLocX, iLocY, nLocY)
        u0 = u0[:, iLocX:iLocX+nLocX, iLocY:iLocY+nLocY]

        MPI.COMM_WORLD.Barrier()
        sleep(0.01*MPI_RANK)
        print(f"[Rank {MPI_RANK}] pRankX={pRankX} ({iLocX}, {nLocX}), pRankY={pRankY} ({iLocY}, {nLocY})")
        MPI.COMM_WORLD.Barrier()

        f1 = Cart2D(dType, fileName)
        f1.setHeader(nVar=u0.shape[0], gridX=x, gridY=y)

    u0 = np.asarray(u0, dtype=f1.dtype)
    f1.initialize()

    nTimes = 1000
    if MPI_RANK == 0:
        print("Starting computations ...")
        tBeg = time()
    for t in np.arange(nTimes)/nTimes:
        f1.addField(t, t*u0)
    if MPI_RANK == 0:
        print(f" -> done in {time()-tBeg:1.4f}s !")

    f2 = FieldsIO.fromFile(fileName)
    t, u = f2.readField(2)
    if MPI_RANK == 0:
        print(f2)
        # print(f2.header)
        # print(u, t, u.shape)
        # print("times :", f2.times)
    assert np.allclose(u, t*u0)
