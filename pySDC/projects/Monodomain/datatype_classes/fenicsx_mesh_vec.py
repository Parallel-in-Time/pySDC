from pySDC.projects.ExplicitStabilized.datatype_classes.fenicsx_mesh import fenicsx_mesh
import numpy as np
from pySDC.core.Errors import DataError
from dolfinx import fem


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


class fenicsx_mesh_vec(object):
    """
    Vector of FEniCSx Function data type
    """

    def __init__(self, init=None, val=0.0, size=1):
        if isinstance(init, fenicsx_mesh_vec):
            self.val_list = [fenicsx_mesh(init_k) for init_k in init.val_list]
        else:
            self.val_list = [fenicsx_mesh(init, val) for _ in range(size)]
        self.size = len(self.val_list)

        self.np_list = [self.val_list[i].values.x.array for i in range(self.size)]

    def __getitem__(self, key):
        return self.val_list[key]

    def __setitem__(self, key):
        return self.val_list[key]

    def copy(self, other):
        for i in range(self.size):
            self.val_list[i].copy(other[i])

    def __add__(self, other):
        me = fenicsx_mesh_vec(self)
        me += other
        return me

    def __sub__(self, other):
        me = fenicsx_mesh_vec(self)
        me -= other
        return me

    def __mul__(self, other):
        if isinstance(other, fenicsx_mesh_vec):
            V = self.val_list[0].values.function_space
            mult = fenicsx_mesh_vec(init=V, val=0.0, size=self.size)
            for i in range(self.size):
                mult.val_list[i].values.x.array[:] = self.val_list[i].values.x.array[:] * other.val_list[i].values.x.array[:]

            return mult
        else:
            raise DataError("Type error: cannot rmul %s to %s" % (type(other), type(self)))

    def __rmul__(self, other):
        if isinstance(other, float):
            me = fenicsx_mesh_vec(self)
            me *= other
            return me
        else:
            raise DataError("Type error: cannot rmul %s to %s" % (type(other), type(self)))

    def __iadd__(self, other):
        for i in range(self.size):
            self.val_list[i] += other.val_list[i]
        return self

    def __isub__(self, other):
        for i in range(self.size):
            self.val_list[i] -= other.val_list[i]
        return self

    def __imul__(self, other):
        for i in range(self.size):
            self.val_list[i] *= other
        return self

    def __abs__(self):
        l2_norm = 0.0
        for val in self.val_list:
            l2_norm += abs(val) ** 2
        return np.sqrt(l2_norm / self.size)
        # return abs(self.val_list[0])

    def dot(self, other):
        sum = 0.0
        for i in range(self.size):
            self.val_list[i].values.vector.dotBegin(other.val_list[i].values.vector)
            sum += self.val_list[i].values.vector.dotEnd(other.val_list[i].values.vector)
        return sum

    def dot_sub(self, other):
        # sum = []
        # Nx = self.val_list[0].values.x.array.size
        # for n in range(Nx):
        #     sum_tmp=0.
        #     for i in range(self.size):
        #         sum_tmp += self.val_list[i].values.x.array[n]*other.val_list[i].values.x.array[n]
        #     sum.append(sum_tmp)
        sums = np.multiply(self.val_list[0].values.x.array, other.val_list[0].values.x.array)
        for i in range(1, self.size):
            sums += np.multiply(self.val_list[i].values.x.array, other.val_list[i].values.x.array)

        return sums

    def axpy(self, a, x):
        for i in range(self.size):
            self[i].axpy(a, x[i])

    def aypx(self, a, x):
        for i in range(self.size):
            self[i].aypx(a, x[i])

    def axpby(self, a, b, x):
        for i in range(self.size):
            self[i].axpby(a, b, x[i])

    def zero(self):
        for i in range(self.size):
            with self[i].values.vector.localForm() as loc_self:
                loc_self.set(0.0)

    def isend(self, dest=None, tag=None, comm=None):
        req_list = [self.val_list[i].isend(dest, tag, comm) for i in range(self.size)]
        return RequestsList(req_list)

    def irecv(self, source=None, tag=None, comm=None):
        req_list = [self.val_list[i].irecv(source, tag, comm) for i in range(self.size)]
        return RequestsList(req_list)

    def bcast(self, root=None, comm=None):
        for i in range(self.size):
            self.val_list[i].bcast(root, comm)

        return self

    def ghostUpdate(self, addv, mode, all=True):
        if all:
            for i in range(self.size):
                self[i].values.vector.ghostUpdate(addv, mode)
        else:
            self[0].values.vector.ghostUpdate(addv, mode)

    def interpolate(self, other):
        if isinstance(other, fenicsx_mesh_vec):
            if self.size == other.size:
                for i in range(self.size):
                    self.val_list[i].values.interpolate(
                        other.val_list[i].values,
                        nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
                            self.val_list[i].values.function_space.mesh._cpp_object, self.val_list[i].values.function_space.element, other.val_list[i].values.function_space.mesh._cpp_object
                        ),
                    )
            else:
                raise DataError("Size error: interpolating vectors have different sizes.")
        else:
            raise DataError("Type error: cannot interpolate %s to %s" % (type(other), type(self)))

    def sub(self, i):
        return self.val_list[i].values

    def getSize(self):
        return self.val_list[0].values.vector.getSize() * self.size

    @property
    def n_loc_dofs(self):
        return self.val_list[0].values.vector.getSizes()[0] * self.size

    @property
    def n_ghost_dofs(self):
        return self.val_list[0].values.x.array.size * self.size - self.n_loc_dofs

    def iadd_sub(self, other, indices):
        for i in indices:
            self.val_list[i] += other.val_list[i]

    def isub_sub(self, other, indices):
        for i in indices:
            self.val_list[i] -= other.val_list[i]

    def mul_sub(self, other, indices):
        V = self.val_list[0].values.function_space
        mult = fenicsx_mesh_vec(init=V, val=0.0, size=self.size)
        for i in indices:
            mult.val_list[i].values.x.array[:] = self.val_list[i].values.x.array[:] * other.val_list[i].values.x.array[:]

        return mult

    def imul_sub(self, other, indices):
        if isinstance(other, float):
            for i in indices:
                self.val_list[i] *= other
        elif isinstance(other, fenicsx_mesh_vec):
            for i in indices:
                self.val_list[i].values.x.array[:] *= other.val_list[i].values.x.array[:]
        else:
            raise DataError("Type error: cannot multiply %s with %s" % (type(other), type(self)))

    def axpby_sub(self, a, b, x, indices):
        for i in indices:
            self.val_list[i].axpby(a, b, x.val_list[i])

    def axpy_sub(self, a, x, indices):
        for i in indices:
            self.val_list[i].axpy(a, x.val_list[i])

    def aypx_sub(self, a, x, indices):
        for i in indices:
            self.val_list[i].aypx(a, x.val_list[i])

    def copy_sub(self, other, indices):
        for i in indices:
            self.val_list[i].copy(other.val_list[i])

    def zero_sub(self, indices):
        for i in indices:
            self.val_list[i].zero()

    # def swap_sub(self,other,indices):
    #     for i in indices:
    #         self.val_list[i], other.val_list[i] = other.val_list[i], self.val_list[i]


class rhs_fenicsx_mesh_vec(object):
    """
    Vector of rhs FEniCSx Function data type
    """

    def __init__(self, init=None, val=0.0, size=1):
        if isinstance(init, rhs_fenicsx_mesh_vec):
            self.expl = fenicsx_mesh_vec(init.expl)
            self.impl = fenicsx_mesh_vec(init.impl)
            self.size = len(self.expl.val_list)
        else:
            self.expl = fenicsx_mesh_vec(init, val, size)
            self.impl = fenicsx_mesh_vec(init, val, size)
            self.size = size

    # def bcast(self, root=None, comm=None):
    #     self.expl.bcast(root, comm)
    #     self.impl.bcast(root, comm)

    #     return self


class exp_rhs_fenicsx_mesh_vec(object):
    """
    Vector of rhs FEniCSx Function data type
    """

    def __init__(self, init=None, val=0.0, size=1):
        if isinstance(init, exp_rhs_fenicsx_mesh_vec):
            self.expl = fenicsx_mesh_vec(init.expl)
            self.impl = fenicsx_mesh_vec(init.impl)
            self.exp = fenicsx_mesh_vec(init.exp)
            self.size = len(self.expl.val_list)
        else:
            self.expl = fenicsx_mesh_vec(init, val, size)
            self.impl = fenicsx_mesh_vec(init, val, size)
            self.exp = fenicsx_mesh_vec(init, val, size)
            self.size = size

    # def bcast(self, root=None, comm=None):
    #     self.expl.bcast(root, comm)
    #     self.impl.bcast(root, comm)
    #     self.exp.bcast(root, comm)

    #     return self
