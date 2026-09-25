"""A distributed array that lives in device memory."""

from numbers import Number

import cupy as cp
import numpy as np
from mpi4py import MPI
from mpi4py_fft.pencil import Subcomm

from pySDC.helpers.fft._transfer import CuPyPencil


class DistArrayCuPy(cp.ndarray):
    """Part of a global array, held on this rank's GPU.

    The counterpart of :class:`mpi4py_fft.distarray.DistArray`, which is a NumPy subclass and so
    cannot hold device memory. Where that class puts the shared bookkeeping in its own body, this
    keeps it here and inherits the storage from CuPy, so the two stay independent: the CPU path
    goes on using released ``mpi4py-fft`` untouched.

    Which part of the global array this rank holds is decided by a
    :class:`~pySDC.helpers.fft._transfer.CuPyPencil`, and moving between two such decompositions
    is what :meth:`redistribute` does.
    """

    def __new__(
        cls,
        global_shape,
        subcomm=None,
        val=None,
        dtype=float,
        memptr=None,
        strides=None,
        alignment=None,
        rank=0,
    ):
        if len(global_shape[rank:]) < 2:
            # Nothing to distribute over: one axis cannot be split and transformed at once, so
            # this is an ordinary local array that simply answers the same questions.
            obj = cp.ndarray.__new__(cls, global_shape, dtype=dtype, memptr=memptr, strides=strides)
            if memptr is None and isinstance(val, Number):
                obj.fill(val)
            obj._rank = rank
            obj._p0 = None
            return obj

        subcomm = cls.get_subcomm(subcomm, global_shape, rank, alignment)
        p0, subshape = cls.setup_pencil(subcomm, rank, global_shape, alignment)

        obj = cp.ndarray.__new__(cls, subshape, dtype=dtype, memptr=memptr, strides=strides)
        if memptr is None and isinstance(val, Number):
            obj.fill(val)
        obj._p0 = p0
        obj._rank = rank
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._p0 = getattr(obj, '_p0', None)
        self._rank = getattr(obj, '_rank', None)

    @staticmethod
    def get_subcomm(subcomm, global_shape, rank, alignment):
        """Turn whatever the caller gave as a decomposition into a `Subcomm`."""
        if isinstance(subcomm, Subcomm):
            return subcomm

        if isinstance(subcomm, (tuple, list)):
            assert len(subcomm) == len(global_shape[rank:])
            # a tuple of communicators is already a decomposition; a tuple of ints is a request
            if not all(isinstance(s, MPI.Comm) for s in subcomm):
                subcomm = Subcomm(MPI.COMM_WORLD, subcomm)
            return subcomm

        assert subcomm is None
        dims = [0] * len(global_shape[rank:])
        # the aligned axis is the one that must not be split, since it is transformed locally
        dims[alignment if alignment is not None else -1] = 1
        return Subcomm(MPI.COMM_WORLD, dims)

    @classmethod
    def setup_pencil(cls, subcomm, rank, global_shape, alignment):
        """Work out this rank's share of the global array."""
        sizes = [s.Get_size() for s in subcomm]
        if alignment is not None:
            assert isinstance(alignment, (int, np.integer))
            assert sizes[alignment] == 1, f'axis {alignment} is split over {sizes[alignment]} ranks, so nothing aligns'
        else:
            # the last undivided axis, which is the last one that can be transformed locally
            alignment = np.flatnonzero(np.array(sizes) == 1)[-1]

        p0 = CuPyPencil(subcomm, global_shape[rank:], axis=alignment)

        subshape = p0.subshape
        if rank > 0:
            subshape = global_shape[:rank] + subshape
        return p0, subshape

    @property
    def alignment(self):
        """The axis this rank holds whole, and can therefore transform."""
        return self._p0.axis

    @property
    def global_shape(self):
        return self.shape[: self.rank] + self._p0.shape

    @property
    def substart(self):
        return (0,) * self.rank + self._p0.substart

    @property
    def subcomm(self):
        return (MPI.COMM_SELF,) * self.rank + self._p0.subcomm

    @property
    def commsizes(self):
        return [s.Get_size() for s in self.subcomm]

    @property
    def pencil(self):
        return self._p0

    @property
    def rank(self):
        """Tensor rank: the number of leading axes that are components rather than space."""
        return self._rank

    @property
    def dimensions(self):
        return len(self._p0.shape)

    @property
    def v(self):
        """This rank's data as a plain CuPy array."""
        return cp.ndarray.__getitem__(self, slice(None, None, None))

    def get(self, *args, **kwargs):
        """`cupy.ndarray.get` copies to the host; keep that, since nothing here overrides it."""
        return cp.ndarray.get(self, *args, **kwargs)

    def asnumpy(self):
        return self.get()

    def local_slice(self):
        """Where this rank's part sits inside the global array."""
        here = [slice(start, start + length) for start, length in zip(self._p0.substart, self._p0.subshape)]
        return tuple([slice(0, s) for s in self.shape[: self.rank]] + here)

    def get_pencil_and_transfer(self, axis):
        """The pencil aligned in `axis`, and the transfer that gets the data there."""
        p1 = self._p0.pencil(axis)
        return p1, self._p0.transfer(p1, self.dtype)

    def redistribute(self, axis=None, out=None):
        """Realign this array along `axis`, returning the redistributed array."""
        if axis == self.alignment:
            return self

        if axis is not None and out is not None:
            assert axis == out.alignment

        if axis is not None and self.commsizes[self.rank + axis] == 1:
            # already whole along that axis, so only the label has to change
            self.pencil.axis = axis
            return self

        if out is not None:
            assert self.global_shape == out.global_shape
            axis = out.alignment
            if self.commsizes == out.commsizes:
                out[:] = self
                return out
            for i in range(len(self._p0.shape)):
                if i not in (self.alignment, out.alignment):
                    assert self.pencil.subcomm[i] == out.pencil.subcomm[i]
                    assert self.pencil.subshape[i] == out.pencil.subshape[i]

        p1, transfer = self.get_pencil_and_transfer(axis)
        if out is None:
            out = type(self)(self.global_shape, subcomm=p1.subcomm, dtype=self.dtype, alignment=axis, rank=self.rank)

        # a tensor is redistributed one component at a time, since only the trailing axes are split
        if self.rank == 0:
            transfer.forward(self, out)
        elif self.rank == 1:
            for i in range(self.shape[0]):
                transfer.forward(self[i], out[i])
        elif self.rank == 2:
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    transfer.forward(self[i, j], out[i, j])
        else:
            raise NotImplementedError(f'Cannot redistribute a tensor of rank {self.rank}')

        transfer.destroy()
        return out
