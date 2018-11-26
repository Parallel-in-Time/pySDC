from pySDC.core.Errors import DataError
import dolfin as df


class fenics_mesh(object):
    """
    FEniCS Function data type with arbitrary dimensions

    Attributes:
        values (np.ndarray): contains the ndarray of the values
    """

    def __init__(self, init=None, val=0.0):
        """
        Initialization routine

        Args:
            init: can either be a FunctionSpace or another fenics_mesh object
            val: initial value (default: 0.0)
        Raises:
            DataError: if init is none of the types above
        """
        # if init is another fenic_mesh, do a deepcopy (init by copy)
        if isinstance(init, type(self)):
            self.values = init.values.copy(deepcopy=True)
        # if init is FunctionSpace, create mesh object with val as initial value
        elif isinstance(init, df.Function):
            self.values = init.copy(deepcopy=True)
        elif isinstance(init, df.FunctionSpace):
            self.values = df.Function(init)
            self.values.vector()[:] = val
        else:
            raise DataError('something went wrong during %s initialization' % type(init))

    def __add__(self, other):
        """
        Overloading the addition operator for mesh types

        Args:
            other (fenics_mesh): mesh object to be added
        Raises:
            DataError: if other is not a mesh object
        Returns:
            fenics_mesh: sum of caller and other values (self+other)
        """

        if isinstance(other, type(self)):
            # always create new mesh, since otherwise c = a + b changes a as well!
            me = fenics_mesh(other)
            me.values.vector()[:] = self.values.vector()[:] + other.values.vector()[:]
            return me
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))

    def __sub__(self, other):
        """
        Overloading the subtraction operator for mesh types

        Args:
            other (fenics_mesh): mesh object to be subtracted
        Raises:
            DataError: if other is not a mesh object
        Returns:
            fenics_mesh: differences between caller and other values (self-other)
        """

        if isinstance(other, type(self)):
            # always create new mesh, since otherwise c = a - b changes a as well!
            me = fenics_mesh(other)
            me.values.vector()[:] = self.values.vector()[:] - other.values.vector()[:]
            return me
        else:
            raise DataError("Type error: cannot subtract %s from %s" % (type(other), type(self)))

    def __rmul__(self, other):
        """
        Overloading the right multiply by factor operator for mesh types

        Args:
            other (float): factor
        Raises:
            DataError: is other is not a float
        Returns:
            fenics_mesh: copy of original values scaled by factor
        """

        if isinstance(other, float):
            # always create new mesh, since otherwise c = f*a changes a as well!
            me = fenics_mesh(self)
            me.values.vector()[:] = other * self.values.vector()[:]
            return me
        else:
            raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))

    def __abs__(self):
        """
        Overloading the abs operator for mesh types

        Returns:
            float: absolute maximum of all mesh values
        """

        # take absolute values of the mesh values

        absval = df.norm(self.values, 'L2')

        # return maximum
        return absval


class rhs_fenics_mesh(object):
    """
    RHS data type for fenics_meshes with implicit and explicit components

    This data type can be used to have RHS with 2 components (here implicit and explicit)

    Attributes:
        impl (fenics_mesh): implicit part
        expl (fenics_mesh): explicit part
    """

    def __init__(self, init, val=0.0):
        """
        Initialization routine

        Args:
            init: can either be a tuple (one int per dimension) or a number (if only one dimension is requested)
                  or another rhs_imex_mesh object
            val (float): an initial number (default: 0.0)
        Raises:
            DataError: if init is none of the types above
        """
        # if init is another rhs_imex_mesh, do a deepcopy (init by copy)
        if isinstance(init, type(self)):
            self.impl = fenics_mesh(init.impl)
            self.expl = fenics_mesh(init.expl)
        # if init is a number or a tuple of numbers, create mesh object with None as initial value
        elif isinstance(init, df.FunctionSpace):
            self.impl = fenics_mesh(init, val=val)
            self.expl = fenics_mesh(init, val=val)
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))

    def __sub__(self, other):
        """
        Overloading the subtraction operator for rhs types

        Args:
            other (rhs_fenics_mesh): rhs object to be subtracted
        Raises:
            DataError: if other is not a rhs object
        Returns:
            rhs_fenics_mesh: differences between caller and other values (self-other)
        """

        if isinstance(other, rhs_fenics_mesh):
            # always create new rhs_imex_mesh, since otherwise c = a - b changes a as well!
            me = rhs_fenics_mesh(self)
            me.impl = self.impl - other.impl
            me.expl = self.expl - other.expl
            return me
        else:
            raise DataError("Type error: cannot subtract %s from %s" % (type(other), type(self)))

    def __add__(self, other):
        """
         Overloading the addition operator for rhs types

        Args:
            other (rhs_fenics_mesh): rhs object to be added
        Raises:
            DataError: if other is not a rhs object
        Returns:
            rhs_fenics_mesh: sum of caller and other values (self-other)
        """

        if isinstance(other, rhs_fenics_mesh):
            # always create new rhs_imex_mesh, since otherwise c = a + b changes a as well!
            me = rhs_fenics_mesh(self)
            me.impl = self.impl + other.impl
            me.expl = self.expl + other.expl
            return me
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))

    def __rmul__(self, other):
        """
        Overloading the right multiply by factor operator for mesh types

        Args:
            other (float): factor
        Raises:
            DataError: is other is not a float
        Returns:
            rhs_fenics_mesh: copy of original values scaled by factor
        """

        if isinstance(other, float):
            # always create new mesh, since otherwise c = f*a changes a as well!
            me = rhs_fenics_mesh(self)
            me.impl = other * self.impl
            me.expl = other * self.expl
            return me
        else:
            raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))
