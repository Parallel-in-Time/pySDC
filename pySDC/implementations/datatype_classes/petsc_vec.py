import numpy as np
from petsc4py import PETSc

from pySDC.implementations.datatype_classes.container import MultiComponentContainer


def _shift(index, start):
    """Turn one global grid index, as petsc4py's ``getVecArray`` takes it, into a local one."""
    if type(index) is int:
        return index - start
    if isinstance(index, slice):
        return slice(
            None if index.start is None else index.start - start,
            None if index.stop is None else index.stop - start,
            index.step,
        )
    return index if index is None or index is Ellipsis else index - start


def _dmda_layout(da, size):
    """
    Starts, shape and strides (in elements) of a vector with local size ``size`` on ``da``: owned points, or owned and
    ghost points. Cached on the DMDA, since the problem classes build arrays in their innermost loops.
    """
    layouts = da.getAttr('pySDC_layouts')
    if layouts is None:
        layouts = {}
        da.setAttr('pySDC_layouts', layouts)
    if size not in layouts:
        dof = da.getDof()
        for ranges in [da.getRanges(), da.getGhostRanges()]:
            sizes = [end - start for start, end in ranges]
            if size == dof * np.prod(sizes):
                break
        else:
            raise ValueError(f'Vector local size {size} is not compatible with the DMDA local sizes')
        # the components of a point are contiguous, and the first grid index runs fastest
        strides = [dof * int(np.prod(sizes[:i])) for i in range(len(sizes))]
        if dof > 1:
            sizes, strides = sizes + [dof], strides + [1]
        layouts[size] = (tuple(start for start, _ in ranges), tuple(sizes), tuple(strides))
    return layouts[size]


class DMDAArray:
    """
    The local part of a vector on a ``DMDA``, indexed by global grid indices with the components last, as
    ``DMDA.getVecArray`` gives it. petsc4py builds that by setting the shape and strides of a NumPy array, which
    NumPy deprecated in 2.4 and 2.5, so this makes a strided view instead. Like petsc4py, it holds the vector's buffer
    until the array is released, which is when PETSc learns of the writes.
    """

    __slots__ = ('array', 'starts', '_ndim')

    def __init__(self, da, vec):
        self.starts, shape, strides = _dmda_layout(da, vec.getLocalSize())
        buffer = np.asarray(vec.getBuffer())
        self.array = np.ndarray(shape, buffer.dtype, buffer, 0, tuple(s * buffer.itemsize for s in strides))
        self._ndim = len(self.starts)

    def _local(self, index):
        if type(index) is tuple:
            return tuple(map(_shift, index, self.starts)) + index[self._ndim :]
        return _shift(index, self.starts[0])

    def __getitem__(self, index):
        return self.array[self._local(index)]

    def __setitem__(self, index, value):
        self.array[self._local(index)] = value


class petsc_vec(PETSc.Vec):
    """PETSc ``Vec`` datatype, created from a ``DMDA`` or copied from another vector."""

    __array_priority__ = 1000  # otherwise rmul with float64 does not work (don't ask, won't tell)

    def __new__(cls, init=None, val=0.0):
        if isinstance(init, petsc_vec) or isinstance(init, PETSc.Vec):
            obj = PETSc.Vec.__new__(cls)
            init.copy(obj)
        elif isinstance(init, PETSc.DMDA):
            tmp = init.createGlobalVec()
            obj = petsc_vec(tmp)
            objarr = DMDAArray(init, obj)
            objarr[:] = val
        else:
            obj = PETSc.Vec.__new__(cls)
        return obj

    def __abs__(self):
        """
        Overloading the abs operator

        Returns:
            float: absolute maximum of all vec values
        """
        # take absolute values of the mesh values (INF = 3)
        return self.norm(3)

    def isend(self, dest=None, tag=None, comm=None):
        """
        Routine for sending data forward in time (non-blocking)

        Args:
            dest (int): target rank
            tag (int): communication tag
            comm: communicator

        Returns:
            request handle
        """
        return comm.Issend(self.getArray(), dest=dest, tag=tag)

    def irecv(self, source=None, tag=None, comm=None):
        """
        Routine for receiving in time

        Args:
            source (int): source rank
            tag (int): communication tag
            comm: communicator

        Returns:
            None
        """
        return comm.Irecv(self.getArray(), source=source, tag=tag)

    def bcast(self, root=None, comm=None):
        """
        Routine for broadcasting values

        Args:
            root (int): process with value to broadcast
            comm: communicator

        Returns:
            broadcasted values
        """
        comm.Bcast(self.getArray(), root=root)
        return self


class petsc_vec_imex(MultiComponentContainer):
    """
    RHS data type for Vec with implicit and explicit components

    Attributes:
        impl (petsc_vec): implicit part
        expl (petsc_vec): explicit part
    """

    components = ['impl', 'expl']
    component_type = petsc_vec


class petsc_vec_comp2(MultiComponentContainer):
    """
    RHS data type for Vec with two components

    Attributes:
        comp1 (petsc_vec): first component
        comp2 (petsc_vec): second component
    """

    components = ['comp1', 'comp2']
    component_type = petsc_vec
