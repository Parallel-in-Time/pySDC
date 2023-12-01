import numpy as np
from pySDC.core.Errors import DataError


class RequestsList:
    def __init__(self, req_list):
        self.req_list = req_list
        self.size = len(req_list)

    def Test(self):
        return all([req.Test() for req in self.req_list])

    def Wait(self):
        for req in self.req_list:
            req.Wait()
        return True


class VectorOfVectors(object):
    def __init__(self, init=None, val=0.0, type_sub_vector=None, size=1):
        if isinstance(init, VectorOfVectors):
            self.val_list = [init_k.copy() for init_k in init.val_list]
            self.type_sub_vector = init.type_sub_vector
        else:
            if isinstance(val, list):
                self.val_list = [type_sub_vector(init, val[i]) for i in range(size)]
            else:
                self.val_list = [type_sub_vector(init, val) for _ in range(size)]
            self.type_sub_vector = type_sub_vector

        self.size = len(self.val_list)

        self.np_list = [self.val_list[i].get_numpy_array() for i in range(self.size)]

    def __getitem__(self, key):
        return self.val_list[key]

    def __setitem__(self, key):
        return self.val_list[key]

    def copy(self, other=None):
        if other is None:
            return VectorOfVectors(self)
        else:
            for i in range(self.size):
                self[i].copy(other[i])

    def __add__(self, other):
        me = VectorOfVectors(self)
        me += other
        return me

    def __sub__(self, other):
        me = VectorOfVectors(self)
        me -= other
        return me

    def __mul__(self, other):
        if isinstance(other, VectorOfVectors) or isinstance(other, float):
            me = VectorOfVectors(self)
            me *= other
            return me
        else:
            raise DataError("Type error: cannot mul %s to %s" % (type(other), type(self)))

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
        elif isinstance(other, VectorOfVectors):
            for i in range(self.size):
                # self.val_list[i] *= other.val_list[i] # dont know why but this does not work, makes self.val_list[i] be None
                self.val_list[i].values.x.array[:] *= other.val_list[i].values.x.array[:]
        else:
            raise DataError("Type error: cannot imul %s to %s" % (type(other), type(self)))
        return self

    def __abs__(self):
        return np.sqrt(np.sum([abs(val) ** 2 for val in self.val_list]) / self.size)

    def dot(self, other):
        return np.sum([self[i].dot(other[i]) for i in range(self.size)])

    def axpy(self, a, x):
        [self[i].axpy(a, x[i]) for i in range(self.size)]

    def aypx(self, a, x):
        [self[i].aypx(a, x[i]) for i in range(self.size)]

    def axpby(self, a, b, x):
        [self[i].axpby(a, b, x[i]) for i in range(self.size)]

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
        if all:
            [self[i].ghostUpdate(addv, mode) for i in range(self.size)]
        else:
            self[0].ghostUpdate(addv, mode)

    def interpolate(self, other):
        if isinstance(other, VectorOfVectors):
            for i in range(self.size):
                self[i].interpolate(other[i])
        else:
            raise DataError("Type error: cannot interpolate %s to %s" % (type(other), type(self)))

    def restrict(self, other):
        if isinstance(other, VectorOfVectors):
            for i in range(self.size):
                self[i].restrict(other[i])
        else:
            raise DataError("Type error: cannot restrict %s to %s" % (type(other), type(self)))

    def prolong(self, other):
        if isinstance(other, VectorOfVectors):
            for i in range(self.size):
                self[i].prolong(other[i])
        else:
            raise DataError("Type error: cannot prolong %s to %s" % (type(other), type(self)))

    @property
    def n_loc_dofs(self):
        return self[0].n_loc_dofs * self.size

    @property
    def n_ghost_dofs(self):
        return self[0].n_ghost_dofs * self.size

    def getSize(self):
        return self[0].getSize() * self.size

    def iadd_sub(self, other, indices):
        for i in indices:
            self.val_list[i] += other[i]

    def isub_sub(self, other, indices):
        for i in indices:
            self.val_list[i] -= other[i]

    def imul_sub(self, other, indices):
        if isinstance(other, float):
            for i in indices:
                self.val_list[i] *= other
        elif isinstance(other, VectorOfVectors):
            for i in indices:
                self.val_list[i] *= other[i]
        else:
            raise DataError("Type error: cannot multiply %s with %s" % (type(other), type(self)))

    def axpby_sub(self, a, b, x, indices):
        [self[i].axpby(a, b, x[i]) for i in indices]

    def axpy_sub(self, a, x, indices):
        [self[i].axpy(a, x[i]) for i in indices]

    def aypx_sub(self, a, x, indices):
        [self[i].aypx(a, x[i]) for i in indices]

    def copy_sub(self, other, indices):
        [self[i].copy(other[i]) for i in indices]

    def zero_sub(self, indices):
        [self[i].zero() for i in indices]


class IMEXEXP_VectorOfVectors(object):
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

    def ghostUpdate(self, addv, mode, all=True):
        self.impl.ghostUpdate(addv, mode, all)
        self.expl.ghostUpdate(addv, mode, all)
        self.exp.ghostUpdate(addv, mode, all)

    def interpolate(self, other):
        self.impl.interpolate(other.impl)
        self.expl.interpolate(other.expl)
        self.exp.interpolate(other.exp)

    def restrict(self, other):
        self.impl.restrict(other.impl)
        self.expl.restrict(other.expl)
        self.exp.restrict(other.exp)

    def prolong(self, other):
        self.impl.prolong(other.impl)
        self.expl.prolong(other.expl)
        self.exp.prolong(other.exp)
