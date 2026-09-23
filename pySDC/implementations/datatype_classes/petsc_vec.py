from petsc4py import PETSc

from pySDC.implementations.datatype_classes.container import MultiComponentContainer


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
