from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh

import dolfin as df

mesh1 = df.UnitSquareMesh(2, 2)
mesh2 = df.UnitSquareMesh(2, 2)
V1 = df.FunctionSpace(mesh1, "CG", 1)
V2 = df.FunctionSpace(mesh2, "CG", 1)


u1 = df.Function(V1)
v1 = df.TestFunction(V1)
u2 = df.Function(V2)
v2 = df.TestFunction(V2)

u3 = df.Function(V2)
u1.vector()[0] = 1
u3.vector()[:] = u1.vector()[:]
u1.vector()[0] = 2
print(u1.vector()[0], u3.vector()[0])

df.assemble(-1.0 * df.inner(df.grad(u1), df.grad(v1)) * df.dx)
df.assemble(-1.0 * df.inner(df.grad(u2), df.grad(v2)) * df.dx)
df.assemble(-1.0 * df.inner(df.grad(u3), df.grad(v2)) * df.dx)
