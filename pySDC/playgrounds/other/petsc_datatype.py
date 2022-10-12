from petsc4py import PETSc
import numpy as np


class mymesh(PETSc.Vec):
    __array_priority__ = 1000

    def __new__(cls, init=None, val=0.0):
        if isinstance(init, mymesh) or isinstance(init, PETSc.Vec):
            obj = PETSc.Vec().__new__(cls)
            init.copy(obj)
        elif isinstance(init, PETSc.DMDA):
            tmp = init.createGlobalVector()
            obj = mymesh(tmp)
            objarr = init.getVecArray(obj)
            objarr[:] = val
        else:
            obj = PETSc.Vec().__new__(cls)
        return obj

    # def __rmul__(self, other):
    #     print('here')
    #     tmp = self.getArray()
    #     tmp[:] *= other
    #     return self

    def __abs__(self):
        """
        Overloading the abs operator

        Returns:
            float: absolute maximum of all mesh values
        """
        # take absolute values of the mesh values
        # print(self.norm(3))
        return self.norm(3)

    # def __array_finalize__(self, obj):
    #     """
    #     Finalizing the datatype. Without this, new datatypes do not 'inherit' the communicator.
    #     """
    #     if obj is None:
    #         return
    #     self.part1 = getattr(obj, 'part1', None)
    #     self.part2 = getattr(obj, 'part2', None)


# u = mymesh((16, PETSc.COMM_WORLD), val=9)
# uarr = u.getArray()
# # uarr[:] = np.random.rand(4,4)
# print(uarr, u.getOwnershipRanges())
# v = mymesh(u)
# varr = v.getArray()
# print(varr, v.getOwnershipRange() )
# if v.comm.getRank() == 0:
#     uarr[0] = 7
#
# print(uarr)
# print(varr)
#
# w = u + v
# warr = w.getArray()
# print(warr, w.getOwnershipRange())

da = PETSc.DMDA().create([4, 4], stencil_width=1)

u = mymesh(da, val=9.0)
uarr = u.getArray()

v = mymesh(u)
varr = v.getArray()
if v.comm.getRank() == 0:
    uarr[0] = 7
uarr = da.getVecArray(u)
print(uarr[:], uarr.shape, type(u))
print(varr[:], v.getOwnershipRange(), type(v))
exit()
w = np.float(1.0) * (u + v)
warr = da.getVecArray(w)
print(warr[:], da.getRanges(), type(w))

print(w.norm(3))
print(abs(w))

# v = PETSc.Vec().createMPI(16)
# varr = v.getArray()
# varr[:] = 9.0
# a = np.float64(1.0) * v
# print(type(a))


exit()
