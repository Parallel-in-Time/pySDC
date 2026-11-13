from functools import cache

import firedrake as fd

from pySDC.core.errors import DataError
from pySDC.implementations.datatype_classes.container import MultiComponentContainer
from pySDC.helpers.firedrake_ensemble_communicator import FiredrakeEnsembleCommunicator


@cache
def riesz_map(V):
    """
    The L2 Riesz map of a primal space, one per space so its mass matrix is factored once.

    Args:
        V (firedrake.FunctionSpace): The primal space

    Returns:
        firedrake.RieszMap: Maps a cofunction r to the function w with (w, v) = r(v) for all v in `V`
    """
    return fd.RieszMap(V, 'L2', constant_jacobian=True)


class firedrake_mesh(object):
    """
    Wrapper for Firedrake data, either a ``Function`` (primal) or a ``Cofunction`` (dual).

    Dual data are load vectors such as an assembled right-hand side :math:`R(u)` or :math:`M u`, which the
    mass-matrix sweepers work with. Whether an instance is primal or dual is decided by the data it holds, so
    arithmetic keeps the kind of its operands, and combining a primal with a dual instance raises a `DataError`.
    Use :class:`firedrake_dual_mesh` to create dual data from a primal space.

    Attributes:
        functionspace (firedrake.Function or firedrake.Cofunction): firedrake data
    """

    def __init__(self, init, val=0.0):
        """
        Args:
            init (firedrake.FunctionSpace, its dual, firedrake.Function, firedrake.Cofunction or firedrake_mesh):
                Space to create data on, or data to copy
            val (float): Value to assign if `init` is a space, ignored otherwise

        Raises:
            DataError: If `init` has none of these types
        """
        if isinstance(init, fd.functionspaceimpl.FiredrakeDualSpace):
            self.functionspace = fd.Cofunction(init)
            self.functionspace.assign(val)
        elif fd.functionspaceimpl.WithGeometry in type(init).__mro__:
            self.functionspace = fd.Function(init)
            self.functionspace.assign(val)
        elif isinstance(init, fd.Function):
            self.functionspace = fd.Function(init)
        elif isinstance(init, (fd.Cofunction, firedrake_mesh)):
            self.functionspace = getattr(init, 'functionspace', init).copy(deepcopy=True)
        else:
            raise DataError('something went wrong during %s initialization' % type(init))

    def __getattr__(self, key):
        return getattr(self.functionspace, key)

    @property
    def is_dual(self):
        """Whether this holds a ``Cofunction``, i.e. a load vector, rather than a ``Function``."""
        return isinstance(self.functionspace, fd.Cofunction)

    @property
    def asnumpy(self):
        """
        Get a numpy array of the values associated with this data
        """
        return self.functionspace.dat.data

    def _check_operand(self, other):
        """
        Make sure `other` can be added to or subtracted from this instance.

        Raises:
            DataError: If `other` is no `firedrake_mesh`, or one of the other kind (primal or dual)
        """
        if not isinstance(other, firedrake_mesh):
            raise DataError("Type error: cannot combine %s with %s" % (type(other), type(self)))
        if other.is_dual != self.is_dual:
            raise DataError('Cannot combine primal data (a Function) with dual data (a Cofunction)')

    def _axpy(self, alpha, other):
        """
        Return a copy of this plus `alpha` times `other`, with `alpha` 1 or -1.

        Dual data are combined on their PETSc vectors: ``Cofunction.assign`` of a sum goes through a symbolic UFL
        ``FormSum`` and costs about ten times as much as the same for a ``Function``.
        """
        self._check_operand(other)
        me = firedrake_mesh(self)
        if self.is_dual:
            with me.dat.vec as x, other.dat.vec_ro as y:
                x.axpy(alpha, y)
        elif alpha == 1.0:
            me.functionspace.assign(self.functionspace + other.functionspace)
        else:
            me.functionspace.assign(self.functionspace - other.functionspace)
        return me

    def __add__(self, other):
        return self._axpy(1.0, other)

    def __sub__(self, other):
        return self._axpy(-1.0, other)

    def __rmul__(self, other):
        """
        Overloading the right multiply by scalar factor

        Args:
            other (float): factor
        Raises:
            DataError: if other is not a float
        Returns:
            firedrake_mesh: copy of original values scaled by factor
        """

        try:
            me = firedrake_mesh(self)
            if self.is_dual:
                with me.dat.vec as x:
                    x.scale(other)
            else:
                me.functionspace.assign(other * self.functionspace)
            return me
        except TypeError as e:
            raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self))) from e

    def __abs__(self):
        r"""
        Overloading the abs operator for mesh types

        Primal data have the :math:`L^2` norm. Dual data :math:`r` have the norm dual to it,
        :math:`\sqrt{r^T M^{-1} r}`, which is the :math:`L^2` norm of the function :math:`w` with
        :math:`(w, v) = r(v)` for all test functions :math:`v`. Unlike a norm of the vector entries, it does not
        shrink with the mesh width.

        Returns:
            float: The norm
        """
        if self.is_dual:
            representer = riesz_map(self.functionspace.function_space().dual())(self.functionspace)
            return abs(self.functionspace.dat.inner(representer.dat)) ** 0.5
        return fd.norm(self.functionspace, 'L2')

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
        assert (
            type(comm) == FiredrakeEnsembleCommunicator
        ), f'Need to give a FiredrakeEnsembleCommunicator here, not {type(comm)}'
        return comm.Isend(self.functionspace, dest=dest, tag=tag)

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
        assert (
            type(comm) == FiredrakeEnsembleCommunicator
        ), f'Need to give a FiredrakeEnsembleCommunicator here, not {type(comm)}'
        return comm.Irecv(self.functionspace, source=source, tag=tag)

    def bcast(self, root=None, comm=None):
        """
        Routine for broadcasting values

        Args:
            root (int): process with value to broadcast
            comm: communicator

        Returns:
            broadcasted values
        """
        assert (
            type(comm) == FiredrakeEnsembleCommunicator
        ), f'Need to give a FiredrakeEnsembleCommunicator here, not {type(comm)}'
        comm.Bcast(self.functionspace, root=root)
        return self


class firedrake_dual_mesh(firedrake_mesh):
    """
    A `firedrake_mesh` that creates dual data from a primal space, for problems whose right-hand side is a load vector.

    Use it as ``dtype_f`` of problems for the mass-matrix sweepers: pySDC creates right-hand sides from the problem's
    primal space, and this puts them on its dual. Copies and arithmetic behave as for `firedrake_mesh`.
    """

    def __init__(self, init, val=0.0):
        """
        Args:
            init: As for `firedrake_mesh`, except that a primal space is replaced by its dual
            val (float): Value to assign if `init` is a space, ignored otherwise
        """
        if fd.functionspaceimpl.WithGeometry in type(init).__mro__:
            init = init.dual()
        super().__init__(init, val)


class IMEX_firedrake_mesh(MultiComponentContainer):
    """
    Datatype for IMEX integration with firedrake data.

    Attributes:
        impl (firedrake_mesh): implicit part
        expl (firedrake_mesh): explicit part
    """

    components = ['impl', 'expl']
    component_type = firedrake_mesh


class IMEX_firedrake_dual_mesh(IMEX_firedrake_mesh):
    """
    Datatype for IMEX integration with dual firedrake data, see `firedrake_dual_mesh`.

    Attributes:
        impl (firedrake_dual_mesh): implicit part
        expl (firedrake_dual_mesh): explicit part
    """

    component_type = firedrake_dual_mesh
