import dolfin as df

from pySDC.core.errors import DataError
from pySDC.implementations.datatype_classes.container import MultiComponentContainer


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
            me = fenics_mesh(self)
            me += other
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
            me = fenics_mesh(self)
            me -= other
            return me
        else:
            raise DataError("Type error: cannot subtract %s from %s" % (type(other), type(self)))

    def __iadd__(self, other):
        """
        Add another mesh in place, on the vector underneath, without copying through NumPy

        Args:
            other (fenics_mesh): mesh object to be added
        Raises:
            DataError: if other is not a mesh object
        Returns:
            fenics_mesh: the caller, now holding self+other
        """

        if isinstance(other, type(self)):
            self.values.vector().axpy(1.0, other.values.vector())
            return self
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))

    def __isub__(self, other):
        """
        Subtract another mesh in place, on the vector underneath, without copying through NumPy

        Args:
            other (fenics_mesh): mesh object to be subtracted
        Raises:
            DataError: if other is not a mesh object
        Returns:
            fenics_mesh: the caller, now holding self-other
        """

        if isinstance(other, type(self)):
            self.values.vector().axpy(-1.0, other.values.vector())
            return self
        else:
            raise DataError("Type error: cannot subtract %s from %s" % (type(other), type(self)))

    def __rmul__(self, other):
        """
        Overloading the right multiply by factor operator for mesh types

        Args:
            other (float): factor
        Raises:
            DataError: if other is not a float
        Returns:
            fenics_mesh: copy of original values scaled by factor
        """

        if isinstance(other, float):
            # always create new mesh, since otherwise c = f*a changes a as well!
            me = fenics_mesh(self)
            vector = me.values.vector()
            vector *= other
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


class rhs_fenics_mesh(MultiComponentContainer):
    """
    RHS data type for fenics_meshes with implicit and explicit components

    Attributes:
        impl (fenics_mesh): implicit part
        expl (fenics_mesh): explicit part
    """

    components = ['impl', 'expl']
    component_type = fenics_mesh
