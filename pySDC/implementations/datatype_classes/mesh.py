import numpy as np
import copy as cp

from pySDC.Errors import DataError

class mesh():
    """
    Mesh data type with arbitrary dimensions

    This data type can be used whenever structured data with a single unknown per point in space is required

    Attributes:
        values: contains the ndarray of the values
    """

    def __init__(self,init=None,val=None):
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
        if isinstance(init,mesh):
            self.values = cp.deepcopy(init.values)
        # if init is a number or a tuple of numbers, create mesh object with val as initial value
        elif isinstance(init,tuple) or isinstance(init,int):
            self.values = np.empty(init,dtype=np.float64)
            self.values[:] = val
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))


    def __add__(self, other):
        """
        Overloading the addition operator for mesh types

        Args:
            other: mesh object to be added
        Raises:
            DataError: if other is not a mesh object
        Returns:
            sum of caller and other values (self+other)
        """

        if isinstance(other, mesh):
            # always create new mesh, since otherwise c = a + b changes a as well!
            me = mesh(np.shape(self.values))
            me.values = self.values + other.values
            return me
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other),type(self)))


    def __sub__(self, other):
        """
        Overloading the subtraction operator for mesh types

        Args:
            other: mesh object to be subtracted
        Raises:
            DataError: if other is not a mesh object
        Returns:
            differences between caller and other values (self-other)
        """

        if isinstance(other, mesh):
            # always create new mesh, since otherwise c = a - b changes a as well!
            me = mesh(np.shape(self.values))
            me.values = self.values - other.values
            return me
        else:
            raise DataError("Type error: cannot subtract %s from %s" % (type(other),type(self)))


    def __rmul__(self, other):
        """
        Overloading the right multiply by factor operator for mesh types

        Args:
            other: float factor
        Raises:
            DataError: is other is not a float
        Returns:
            mesh object, copy of original values scaled by factor
        """

        if isinstance(other, float) or isinstance(other, complex):
            # always create new mesh, since otherwise c = f*a changes a as well!
            me = mesh(np.shape(self.values))
            me.values = self.values*other
            return me
        else:
            raise DataError("Type error: cannot multiply %s to %s" % (type(other),type(self)))


    def __abs__(self):
        """
        Overloading the abs operator for mesh types

        Returns:
            absolute maximum of all mesh values
        """

        # take absolute values of the mesh values
        absval = abs(self.values)
        # return maximum
        return np.amax(absval)

    def apply_mat(self,A):
        """
        Matrix multiplication operator

        Args:
            A: a matrix

        Returns:
            mesh object, component multiplied by the matrix A
        """
        assert A.shape[1] == self.values.shape[0], "ERROR: cannot apply operator %s to %s" %(A,self)

        me = mesh(A.shape[0])
        me.values = A.dot(self.values)

        return me


class rhs_imex_mesh():

    """
    RHS data type for meshes with implicit and explicit components

    This data type can be used to have RHS with 2 components (here implicit and explicit)

    Attributes:
        impl: implicit part
        expl: explicit part
    """

    def __init__(self,init):
        """
        Initialization routine

        Args:
            init: can either be a tuple (one int per dimension) or a number (if only one dimension is requested)
                  or another rhs_imex_mesh object
        Raises:
            DataError: if init is none of the types above
        """

        # if init is another rhs_imex_mesh, do a deepcopy (init by copy)
        if isinstance(init,type(self)):
            self.impl = mesh(init.impl)
            self.expl = mesh(init.expl)
        # if init is a number or a tuple of numbers, create mesh object with None as initial value
        elif isinstance(init,tuple) or isinstance(init,int):
            self.impl = mesh(init)
            self.expl = mesh(init)
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))


    def __sub__(self, other):
        """
        Overloading the subtraction operator for rhs types

        Args:
            other: rhs object to be subtracted
        Raises:
            DataError: if other is not a rhs object
        Returns:
            differences between caller and other values (self-other)
        """

        if isinstance(other, rhs_imex_mesh):
            # always create new rhs_imex_mesh, since otherwise c = a - b changes a as well!
            me = rhs_imex_mesh(np.shape(self.impl.values))
            me.impl.values = self.impl.values - other.impl.values
            me.expl.values = self.expl.values - other.expl.values
            return me
        else:
            raise DataError("Type error: cannot subtract %s from %s" % (type(other),type(self)))


    def __add__(self, other):
        """
         Overloading the addition operator for rhs types

        Args:
            other: rhs object to be added
        Raises:
            DataError: if other is not a rhs object
        Returns:
            sum of caller and other values (self-other)
        """

        if isinstance(other, rhs_imex_mesh):
            # always create new rhs_imex_mesh, since otherwise c = a + b changes a as well!
            me = rhs_imex_mesh(np.shape(self.impl.values))
            me.impl.values = self.impl.values + other.impl.values
            me.expl.values = self.expl.values + other.expl.values
            return me
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other),type(self)))

    def __rmul__(self, other):
        """
        Overloading the right multiply by factor operator for mesh types

        Args:
            other: float factor
        Raises:
            DataError: is other is not a float
        Returns:
            mesh object, copy of original values scaled by factor
        """

        if isinstance(other, float):
            # always create new rhs_imex_mesh
            me = rhs_imex_mesh(np.shape(self.impl.values))
            me.impl.values = other*self.impl.values
            me.expl.values = other*self.expl.values
            return me
        else:
            raise DataError("Type error: cannot multiply %s to %s" % (type(other),type(self)))

    def apply_mat(self,A):
        """
        Matrix multiplication operator

        Args:
            A: a matrix

        Returns:
            rhs_imex_mesh object, each component multiplied by the matrix A
        """

        assert A.shape[1] == self.impl.values.shape[0], "ERROR: cannot apply operator %s to %s" % (A, self.impl)
        assert A.shape[1] == self.expl.values.shape[0], "ERROR: cannot apply operator %s to %s" % (A, self.expl)

        me = rhs_imex_mesh(A.shape[1])
        me.impl.values = A.dot(self.impl.values)
        me.expl.values = A.dot(self.expl.values)

        return me