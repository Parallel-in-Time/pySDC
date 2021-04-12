from pySDC.implementations.datatype_classes.parallel_mesh import parallel_mesh, parallel_imex_mesh
import numpy as np
from mpi4py import MPI

a = parallel_mesh((10, MPI.COMM_SELF, np.dtype('float64')))
a[:] = 0.001 * np.arange(10)

c = parallel_mesh(a)

d = 2.0*c
print(d, c)
print(d.comm, c.comm)


a = parallel_imex_mesh((10, MPI.COMM_SELF, np.dtype('float64')))
a.impl[:] = np.arange(10)
# a.expl[:] = np.arange(10)

b = parallel_imex_mesh((10, MPI.COMM_SELF, np.dtype('float64')))
b.impl[:] = 0.1 * np.arange(10)
# b.expl[:] = 0.1 * np.arange(10)
print(type(a.impl))
c = a + b
d = a.impl + b.impl
print(a.impl, b.impl, c.impl)
print(a.expl, b.expl, c.expl)
print(a.comm, b.comm, c.comm)