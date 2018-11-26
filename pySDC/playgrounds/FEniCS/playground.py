from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh

import dolfin as df

mesh = df.UnitSquareMesh(2, 2)
V = df.FunctionSpace(mesh, "CG", 1)

# u = df.Function(V)
#

#
# v = u.copy(deepcopy=True)
#
# w =
#
# print(v.vector()[:])
# exit()

u = fenics_mesh(init=V, val=4.0)
# u.values.vector()[:] = 1.0
v = fenics_mesh(init=u)
u.values.vector()[:] = 2.0
w = u + v
x = 0.1*w - u
print(abs(x))
