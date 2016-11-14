from pySDC.core.Errors import DataError
import dolfin as df


class fenics_mesh(object):
    """
    Mesh data type with arbitrary dimensions

    This data type can be used whenever structured data with a single unknown per point in space is required

    Attributes:
        values (np.ndarray): contains the ndarray of the values
        V: function space
    """

    def __init__(self, init=None, val=0.0):
        """
        Initialization routine

        Args:
            init: can either be a tuple (one int per dimension) or a number (if only one dimension is requested)
                  or another mesh object
            val: initial value (default: None)
        Raises:
            DataError: if init is none of the types above
        """

        # if init is another mesh, do a deepcopy (init by copy)
        if isinstance(init, fenics_mesh):
            self.values = df.Function(init.V, init.values)
            self.V = init.V
        # if init is a number or a tuple of numbers, create mesh object with val as initial value
        else:
            self.values = df.Function(init)
            self.V = init
        # (FIXME: cannot define type of functionspace)
        # elif isinstance(init, type(df.FunctionSpace)):
        #     self.values = df.Function(init)
        #     self.V = init
        # something is wrong, if none of the ones above hit
        # else:
        #     raise DataError('something went wrong during %s initialization' % type(init))

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
            me.values = df.Function(self.V, self.values.vector() + other.values.vector())
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
            me.values = df.Function(self.V, self.values.vector() - other.values.vector())
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
            me.values = df.Function(self.V, other * self.values.vector())
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
    RHS data type for meshes with implicit and explicit components

    This data type can be used to have RHS with 2 components (here implicit and explicit)

    Attributes:
        impl (fenics_mesh): implicit part
        expl (fenics_mesh): explicit part
    """

    def __init__(self, init):
        """
        Initialization routine

        Args:
            init: can either be a tuple (one int per dimension) or a number (if only one dimension is requested)
                  or another rhs_imex_mesh object
        Raises:
            DataError: if init is none of the types above
        """

        # if init is another rhs_imex_mesh, do a deepcopy (init by copy)
        if isinstance(init, type(self)):
            self.impl = fenics_mesh(init.impl)
            self.expl = fenics_mesh(init.expl)
        # if init is a number or a tuple of numbers, create mesh object with None as initial value
        elif isinstance(init, df.FunctionSpace):
            self.impl = fenics_mesh(init)
            self.expl = fenics_mesh(init)
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
