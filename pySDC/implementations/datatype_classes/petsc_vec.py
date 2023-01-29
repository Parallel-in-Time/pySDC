from petsc4py import PETSc

from pySDC.core.Errors import DataError


class petsc_vec(PETSc.Vec):
    __array_priority__ = 1000  # otherwise rmul with float64 does not work (don't ask, won't tell)

    def __new__(cls, init=None, val=0.0):
        if isinstance(init, petsc_vec) or isinstance(init, PETSc.Vec):
            obj = PETSc.Vec.__new__(cls)
            init.copy(obj)
        elif isinstance(init, PETSc.DMDA):
            tmp = init.createGlobalVector()
            obj = petsc_vec(tmp)
            objarr = init.getVecArray(obj)
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


class petsc_vec_imex(object):
    """
    RHS data type for Vec with implicit and explicit components

    This data type can be used to have RHS with 2 components (here implicit and explicit)

    Attributes:
        impl (petsc_vec): implicit part
        expl (petsc_vec): explicit part
    """

    def __init__(self, init, val=0.0):
        """
        Initialization routine

        Args:
            init: can either be a tuple (one int per dimension) or a number (if only one dimension is requested)
                  or another imex_mesh object
            val (float): an initial number (default: 0.0)
        Raises:
            DataError: if init is none of the types above
        """

        if isinstance(init, type(self)):
            self.impl = petsc_vec(init.impl)
            self.expl = petsc_vec(init.expl)
        elif isinstance(init, PETSc.DMDA):
            self.impl = petsc_vec(init, val=val)
            self.expl = petsc_vec(init, val=val)
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))


class petsc_vec_comp2(object):
    """
    RHS data type for Vec with implicit and explicit components

    This data type can be used to have RHS with 2 components (here implicit and explicit)

    Attributes:
        impl (petsc_vec): implicit part
        expl (petsc_vec): explicit part
    """

    def __init__(self, init, val=0.0):
        """
        Initialization routine

        Args:
            init: can either be a tuple (one int per dimension) or a number (if only one dimension is requested)
                  or another imex_mesh object
            val (float): an initial number (default: 0.0)
        Raises:
            DataError: if init is none of the types above
        """

        if isinstance(init, type(self)):
            self.comp1 = petsc_vec(init.comp1)
            self.comp2 = petsc_vec(init.comp2)
        elif isinstance(init, PETSc.DMDA):
            self.comp1 = petsc_vec(init, val=val)
            self.comp2 = petsc_vec(init, val=val)
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))
