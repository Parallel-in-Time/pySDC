from petsc4py import PETSc


class mymesh(PETSc.Vec):

    def __new__(cls, init=None, val=0.0):
        if isinstance(init, mymesh):
            obj = PETSc.Vec().__new__(cls)
            init.copy(obj)
            # obj.
            # obj[:] = init[:]
            # obj.part1 = obj[0, :]
            # obj.part2 = obj[1, :]
        elif isinstance(init, tuple):
            obj = PETSc.Vec().__new__(cls)
            obj.createMPI(size=init[0], comm=init[1])
            objarr = obj.getArray()
            objarr[:] = val
        else:
            obj = PETSc.Vec().__new__(cls)
        return obj

    # def __array_finalize__(self, obj):
    #     """
    #     Finalizing the datatype. Without this, new datatypes do not 'inherit' the communicator.
    #     """
    #     if obj is None:
    #         return
    #     self.part1 = getattr(obj, 'part1', None)
    #     self.part2 = getattr(obj, 'part2', None)

u = mymesh((10, PETSc.COMM_WORLD), val=9)
uarr = u.getArray()
# uarr[:] = 1
print(uarr, u.getOwnershipRange())
v = mymesh(u)
varr = v.getArray()
print(varr, v.getOwnershipRange() )
if v.comm.getRank() == 0:
    uarr[0] = 7

print(uarr)
print(varr)

w = u + v
warr = w.getArray()
print(warr, w.getOwnershipRange())
exit()


m = mymesh((10, np.dtype('float64')))
m.part1[:] = 1
m.part2[:] = 2
print(m.part1, m.part2)
print(m)
print()

n = mymesh(m)
n.part1[0] = 10
print(n.part1, n.part2)
print(n)
print()

print(o.part1, o.part2)
print(o)
print()
# print(n + m)