import numpy as np
from pySDC.core.Errors import DataError


class RequestsList:
    """
    A wrapper for a list of MPI requests. It is used to wait for all requests to complete.
    In this context it is used to check communication of all subvectors in a VectorOfVectors.
    """

    def __init__(self, req_list):
        self.req_list = req_list
        self.size = len(req_list)

    def Wait(self):
        for req in self.req_list:
            req.Wait()
        return True


class VectorOfVectors(object):
    """
    A wrapper for a list of subvectors. It is used to represent a vector of vectors, for example in the context of a PDE with many unknowns.

    Parameters:
    ----------
        init: A VectorOfVectors to copy from or the size of the subvectors.
        val (optional): Used when init is not a VectorOfVectors. It is a value to initialize the subvectors. Can be a float, a list of subvectors, a list of numpy arrays. Defaults to 0.0.
        type_sub_vector (optional): Used when init is not a VectorOfVectors. The type of the subvectors.
        size (optional): Used when init is not a VectorOfVectors and val is a float. The number of subvectors. All subvectors will have size=init and are initialized to val.

    Attributes:
    ----------
        val_list: A list of subvectors.
        type_sub_vector: The type of the subvectors.
        size: The number of subvectors.
    """

    def __init__(self, init, val=None, type_sub_vector=None, size=1):
        if isinstance(init, VectorOfVectors):
            self.val_list = [init_k.copy() for init_k in init.val_list]
            self.type_sub_vector = init.type_sub_vector
        else:
            if isinstance(val, list):
                assert len(val) == size, 'val must be a list of size %d' % size
                self.val_list = [type_sub_vector(init, val[i]) for i in range(size)]
            else:
                self.val_list = [type_sub_vector(init, val) for _ in range(size)]
            self.type_sub_vector = type_sub_vector

        self.size = len(self.val_list)

    def __getitem__(self, key):
        """
        Returns the subvector at index key.
        """
        return self.val_list[key]

    @property
    def np_list(self):
        return [self.val_list[i].numpy_array for i in range(self.size)]

    def np_array(self, i):
        return self.val_list[i].numpy_array

    # def is_nan_or_inf(self):
    #     return np.any([self.val_list[i].is_nan_or_inf() for i in range(self.size)])

    def __add__(self, other):
        me = VectorOfVectors(self)
        me += other
        return me

    def __sub__(self, other):
        me = VectorOfVectors(self)
        me -= other
        return me

    def __rmul__(self, other):
        if isinstance(other, float):
            me = VectorOfVectors(self)
            me *= other
            return me
        else:
            raise DataError("Type error: cannot rmul %s to %s" % (type(other), type(self)))

    def __iadd__(self, other):
        for i in range(self.size):
            self.val_list[i] += other[i]
        return self

    def __isub__(self, other):
        for i in range(self.size):
            self.val_list[i] -= other[i]
        return self

    def __imul__(self, other):
        if isinstance(other, float):
            for i in range(self.size):
                self.val_list[i] *= other
        else:
            raise DataError("Type error: cannot imul %s to %s" % (type(other), type(self)))
        return self

    def __abs__(self):
        """
        Returns the norm of the vector. We divide by the square root of the number of subvectors.
        Hence a vector of vectors with all subvectors equal to 1 will have a norm of 1.
        """
        return np.sqrt(np.sum([abs(val) ** 2 for val in self.val_list]) / self.size)

    def rel_norm(self, other):
        """
        For every subvector in self, computes its relative norm with respect to the corresponding subvector in other.
        Returns the average of all relative norms.
        This is different with respect to simply compute the relative norm of the whole vector as abs(self) / abs(other), since in that case small subvectors are overwhelmed by large ones.

        Args:
            other: A VectorOfVectors.
        """
        my_norms = np.array([abs(val) for val in self.val_list])
        other_norms = np.array([abs(val) for val in other.val_list])
        return np.average(my_norms / other_norms)

    def aypx(self, a, x):
        [self[i].aypx(a, x[i]) for i in range(self.size)]

    def zero(self):
        [self[i].zero() for i in range(self.size)]

    def isend(self, dest=None, tag=None, comm=None):
        req_list = [self[i].isend(dest, tag, comm) for i in range(self.size)]
        return RequestsList(req_list)

    def irecv(self, source=None, tag=None, comm=None):
        req_list = [self[i].irecv(source, tag, comm) for i in range(self.size)]
        return RequestsList(req_list)

    def bcast(self, root=None, comm=None):
        [self[i].bcast(root, comm) for i in range(self.size)]
        return self

    def ghostUpdate(self, addv, mode, all=True):
        """
        Updates the ghost nodes of the subvectors.
        Args:
            addv, mode: same arguments as in petsc4py.vector.ghostUpdate
            all: If True, updates all subvectors. If False, updates only the first subvector.
                 For the monodomain equation very often only the first subvector is updated, since the others are driven purely by ODEs and dont need ghost nodes updates.
        """
        if all:
            [self[i].ghostUpdate(addv, mode) for i in range(self.size)]
        else:
            self[0].ghostUpdate(addv, mode)

    def getSize(self):
        return self[0].getSize() * self.size

    def iadd_sub(self, other, indices):
        """
        Performs inplace add only for a subset of subvectors.

        Args:
            other: A VectorOfVectors.
            indices: A list of integers. The indices of the subvectors to add.
        """
        for i in indices:
            self.val_list[i] += other[i]

    def imul_sub(self, other, indices):
        """
        Performs inplace multiply only for a subset of subvectors.

        Args:
            other: A VectorOfVectors or a float.
            indices: A list of integers. The indices of the subvectors to multiply.
        """
        if isinstance(other, float):
            for i in indices:
                self.val_list[i] *= other
        elif isinstance(other, VectorOfVectors):
            for i in indices:
                self.val_list[i] *= other[i]
        else:
            raise DataError("Type error: cannot multiply %s with %s" % (type(other), type(self)))

    def axpy_sub(self, a, x, indices):
        """
        Performs inplace a * x + y only for a subset of subvectors.
        """
        [self[i].axpy(a, x[i]) for i in indices]

    def copy_sub(self, other, indices):
        """
        Copies the values of a subset of subvectors from other to self.
        """
        [self[i].copy(other[i]) for i in indices]

    def zero_sub(self, indices):
        """
        Zero a subset of subvectors.
        """
        [self[i].zero() for i in indices]


class IMEXEXP_VectorOfVectors(object):
    """
    A wrapper for a list of three VectorOfVectors. It is used to represent right-hand sides with three terms: implicit, explicit and exponential.

    Args:
        init: A IMEXEXP_VectorOfVectors to copy from or any other argument that can be passed to VectorOfVectors.__init__
        val: Used when init is not a IMEXEXP_VectorOfVectors. It is used to initialize the three VectorOfVectors as in VectorOfVectors.__init__
        type_sub_vector: Similar to val.
        size: Similar to val.

    Attributes:
        impl: A VectorOfVectors.
        expl: A VectorOfVectors.
        exp: A VectorOfVectors.
    """

    def __init__(self, init=None, val=0.0, type_sub_vector=None, size=1):
        if isinstance(init, IMEXEXP_VectorOfVectors):
            self.expl = VectorOfVectors(init.expl)
            self.impl = VectorOfVectors(init.impl)
            self.exp = VectorOfVectors(init.exp)
            self.size = self.expl.size
        else:
            self.expl = VectorOfVectors(init, val, type_sub_vector, size)
            self.impl = VectorOfVectors(init, val, type_sub_vector, size)
            self.exp = VectorOfVectors(init, val, type_sub_vector, size)
            self.size = size

    def __iadd__(self, other):
        self.impl += other.impl
        self.expl += other.expl
        self.exp += other.exp

        return self

    def __sub__(self, other):
        me = IMEXEXP_VectorOfVectors(self)
        me -= other
        return me

    def __isub__(self, other):
        self.expl -= other.expl
        self.impl -= other.impl
        self.exp -= other.exp
        return self

    def __rmul__(self, other):
        if isinstance(other, float):
            me = IMEXEXP_VectorOfVectors(self)
            me *= other
            return me
        else:
            raise DataError("Type error: cannot rmul %s to %s" % (type(other), type(self)))

    def __imul__(self, other):
        if isinstance(other, float):
            self.impl *= other
            self.expl *= other
            self.exp *= other
        elif isinstance(other, IMEXEXP_VectorOfVectors):
            self.impl *= other.impl
            self.expl *= other.expl
            self.exp *= other.exp
        else:
            raise DataError("Type error: cannot imul %s to %s" % (type(other), type(self)))
        return self

    def ghostUpdate(self, addv, mode, all=True):
        self.impl.ghostUpdate(addv, mode, all)
        self.expl.ghostUpdate(addv, mode, all)
        self.exp.ghostUpdate(addv, mode, all)
