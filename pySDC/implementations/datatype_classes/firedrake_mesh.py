import firedrake as fd

from pySDC.core.errors import DataError


class firedrake_mesh(object):
    """
    Wrapper for firedrake function data.

    Attributes:
        functionspace (firedrake.Function): firedrake data
    """

    def __init__(self, init, val=0.0):
        if fd.functionspaceimpl.WithGeometry in type(init).__mro__:
            self.functionspace = fd.Function(init)
            self.functionspace.assign(val)
        elif fd.Function in type(init).__mro__:
            self.functionspace = fd.Function(init)
        elif type(init) == firedrake_mesh:
            self.functionspace = init.functionspace.copy(deepcopy=True)
        else:
            raise DataError('something went wrong during %s initialization' % type(init))

    def __getattr__(self, key):
        return getattr(self.functionspace, key)

    @property
    def asnumpy(self):
        """
        Get a numpy array of the values associated with this data
        """
        return self.functionspace.dat._numpy_data

    def __add__(self, other):
        if isinstance(other, type(self)):
            me = firedrake_mesh(other)
            me.functionspace.assign(self.functionspace + other.functionspace)
            return me
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))

    def __sub__(self, other):
        if isinstance(other, type(self)):
            me = firedrake_mesh(other)
            me.functionspace.assign(self.functionspace - other.functionspace)
            return me
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))

    def __rmul__(self, other):
        """
        Overloading the right multiply by scalar factor

        Args:
            other (float): factor
        Raises:
            DataError: if other is not a float
        Returns:
            fenics_mesh: copy of original values scaled by factor
        """

        if type(other) in [int, float, complex]:
            me = firedrake_mesh(self)
            me.functionspace.assign(other * self.functionspace)
            return me
        else:
            raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))

    def __abs__(self):
        """
        Overloading the abs operator for mesh types

        Returns:
            float: L2 norm
        """

        return fd.norm(self.functionspace, 'L2')


class IMEX_firedrake_mesh(object):
    """
    Datatype for IMEX integration with firedrake data.

    Attributes:
        impl (firedrake_mesh): implicit part
        expl (firedrake_mesh): explicit part
    """

    def __init__(self, init, val=0.0):
        if type(init) == type(self):
            self.impl = firedrake_mesh(init.impl)
            self.expl = firedrake_mesh(init.expl)
        else:
            self.impl = firedrake_mesh(init, val=val)
            self.expl = firedrake_mesh(init, val=val)

    def __add__(self, other):
        me = IMEX_firedrake_mesh(self)
        me.impl = self.impl + other.impl
        me.expl = self.expl + other.expl
        return me

    def __sub__(self, other):
        me = IMEX_firedrake_mesh(self)
        me.impl = self.impl - other.impl
        me.expl = self.expl - other.expl
        return me

    def __rmul__(self, other):
        me = IMEX_firedrake_mesh(self)
        me.impl = other * self.impl
        me.expl = other * self.expl
        return me
