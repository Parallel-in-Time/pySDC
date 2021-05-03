import numpy as np


class mymesh(np.ndarray):

    def __new__(cls, init, val=0.0):
        """
        Instantiates new datatype. This ensures that even when manipulating data, the result is still a mesh.

        Args:
            init: either another mesh or a tuple containing the dimensions, the communicator and the dtype
            val: value to initialize

        Returns:
            obj of type mesh

        """
        if isinstance(init, mymesh):
            obj = np.ndarray.__new__(cls, init.shape, dtype=init.dtype, buffer=None)
            obj[:] = init[:]
            obj.part1 = obj[0, :]
            obj.part2 = obj[1, :]
        elif isinstance(init, tuple):
            obj = np.ndarray.__new__(cls, shape=(2, init[0]), dtype=init[1], buffer=None)
            obj.fill(val)
            obj.part1 = obj[0, :]
            obj.part2 = obj[1, :]
        else:
            raise NotImplementedError(type(init))
        return obj

    def __array_finalize__(self, obj):
        """
        Finalizing the datatype. Without this, new datatypes do not 'inherit' the communicator.
        """
        if obj is None:
            return
        self.part1 = getattr(obj, 'part1', None)
        self.part2 = getattr(obj, 'part2', None)


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