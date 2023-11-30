import numpy as np

from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.core.Errors import DataError


class DAEMesh(object):
    r"""
    Datatype for DAE problems. The solution of the problem can be splitted in the differential part
    and in an algebraic part.

    This data type can be used for the solution of the problem itself as well as for its derivative.

    Parameters
    ----------
    init :
        Initialization for a mesh. It can either be a tuple (one int per dimension) or a number
        (if only one dimension is requested) or another imex_mesh object.
    val : float, optional
        The value that the mesh is wanted to fill with. Default is ``0.0``.

    Attributes
    ----------
    diff : mesh.mesh
        Differential part.
    alg : mesh.mesh
        Algebraic part.
    """

    def __init__(self, init, val=0.0):
        """Initialization routine"""

        if isinstance(init, type(self)):
            self.diff = mesh(init.diff)
            self.alg = mesh(init.alg)
        elif (
            isinstance(init, tuple)
            and (init[1] is None or str(type(init[1])) == "MPI.Intracomm")
            and isinstance(init[2], np.dtype)
        ):
            self.diff = mesh((init[0][0], init[1], init[2]), val=val)
            self.alg = mesh((init[0][1], init[1], init[2]), val=val)

        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))

    def __abs__(self):
        """
        It chooses the maximum value between the differential part and the algebraic part. If the
        problem contains no algebraic part, then the maximum values is computed over the differential parts.
        """
        return max([abs(self.diff), abs(self.alg)]) if len(self.alg) > 0 else max([abs(self.diff)])

    def __add__(self, other):
        """
        Overloading the addition operator for DAE meshes.
        """

        if isinstance(other, type(self)):
            me = DAEMesh(self)
            me.diff[:] = self.diff + other.diff
            me.alg[:] = self.alg + other.alg
            return me
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))

    def __sub__(self, other):
        """
        Overloading the subtraction operator for DAE meshes.
        """

        if isinstance(other, type(self)):
            me = DAEMesh(self)
            me.diff[:] = self.diff - other.diff
            me.alg[:] = self.alg - other.alg
            return me
        else:
            raise DataError("Type error: cannot subtract %s to %s" % (type(other), type(self)))

    def __rmul__(self, other):
        """
        Overloading the right multiply by factor operator for DAE meshes.
        """

        if isinstance(other, float):
            me = DAEMesh(self)
            me.diff[:] = other * self.diff
            me.alg[:] = other * self.alg
            return me
        else:
            raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))

    def __str__(self):
        """
        Overloading the string operator for DAE Meshes s.t. an output can be separated into the
        differential part and the algebraic part.
        """
        return f"({self.diff}, {self.alg})"
