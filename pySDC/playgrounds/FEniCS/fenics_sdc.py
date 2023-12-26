import dolfin as df
import numpy as np
from pySDC.core.Collocation import CollBase
from pySDC.core.Sweeper import sweeper as Sweeper

c_nvars = 128
refinements = 1
family = 'CG'
order = 4
nu = 0.1

t0 = 0.0
dt = 0.2
nnodes = 3
quad_type = 'RADAU-RIGHT'
node_type = 'LEGENDRE'
kmax = 10
tol = 1E-10
qd_type = 'LU'


# define the Dirichlet boundary
def Boundary(x, on_boundary):
    return on_boundary

def uexact(t, order, V):
    return df.interpolate(df.Expression('cos(a*x[0]) * cos(t)', a=np.pi, t=t, degree=order), V)

# set solver and form parameters
df.parameters["form_compiler"]["optimize"] = True
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters['allow_extrapolation'] = True

# set mesh and refinement (for multilevel)
mesh = df.UnitIntervalMesh(c_nvars)
for _ in range(refinements):
    mesh = df.refine(mesh)

# define function space for future reference
V = df.FunctionSpace(mesh, family, order)
tmp = df.Function(V)
print('DoFs on this level:', len(tmp.vector()[:]))

# set boundary values
# bc = df.DirichletBC(V, df.Constant(1.0), Boundary)
bh = df.DirichletBC(V, df.Constant(0.0), Boundary)

# Stiffness term (Laplace)
u_trial = df.TrialFunction(V)
v_test = df.TestFunction(V)
a_K = -1.0 * df.inner(df.nabla_grad(u_trial), nu * df.nabla_grad(v_test)) * df.dx

# Mass term
a_M = u_trial * v_test * df.dx

M = df.assemble(a_M)
K = df.assemble(a_K)

# # set forcing term as expression
# g = df.Expression(
#     '-sin(a*x[0]) * (sin(t) - b*a*a*cos(t))',
#     a=np.pi,
#     b=nu,
#     t=t0,
#     degree=order,
# )

# Time-dependent boundary conditions
g = df.Expression(
    '-cos(a*x[0]) * (sin(t) - b*a*a*cos(t))',
    a=np.pi,
    b=nu,
    t=t0,
    degree=order,
)

params = {'num_nodes': nnodes, 'quad_type': quad_type, 'node_type': node_type}
sweeper = Sweeper(params)
Q = sweeper.coll.Qmat
QI = sweeper.get_Qdelta_implicit(sweeper.coll, qd_type)
QE = sweeper.get_Qdelta_explicit(sweeper.coll, 'EE')

u = [df.Function(V) for _ in range(nnodes + 1)]
fimpl = [df.Function(V) for _ in range(nnodes + 1)]
fexpl = [df.Function(V) for _ in range(nnodes + 1)]
int = [df.Function(V) for _ in range(nnodes)]
res = [df.Function(V) for _ in range(nnodes)]
u[0] = uexact(t0, order, V)

for m in range(nnodes + 1):
    K.mult(u[m].vector(), fimpl[m].vector())
    if m == 0:
        g.t = t0
    else:
        g.t = t0 + dt * sweeper.coll.nodes[m - 1]
    M.mult(df.interpolate(g, V).vector(), fexpl[m].vector())

for k in range(kmax):

    res_norm = []
    for m in range(nnodes):
        int[m] = df.Function(V)
        M.mult(u[0].vector(), int[m].vector())
        for j in range(nnodes + 1):
            int[m] += dt * Q[m + 1, j] * (fimpl[j] + fexpl[j]) - dt * (QI[m + 1, j] * fimpl[j] + QE[m + 1, j] * fexpl[j])

    for m in range(nnodes):
        rhs = int[m]
        for j in range(m + 1):
            rhs += dt * (QI[m + 1, j] * fimpl[j] + QE[m + 1, j] * fexpl[j])
        rhs = df.project(rhs, V)

        T = M - dt * QI[m + 1, m + 1] * K
        bc = df.DirichletBC(V, df.Expression('cos(a*x[0]) * cos(t)', a=np.pi, t=t0 + dt * sweeper.coll.nodes[m], degree=order), Boundary)
        bc.apply(T, rhs.vector())
        df.solve(T, u[m + 1].vector(), rhs.vector())

        K.mult(u[m + 1].vector(), fimpl[m + 1].vector())
        g.t = t0 + dt * sweeper.coll.nodes[m]
        M.mult(df.interpolate(g, V).vector(), fexpl[m + 1].vector())

    for m in range(nnodes):
        res = df.Function(V)
        M.mult(u[m + 1].vector() - u[0].vector(), res.vector())
        for j in range(nnodes + 1):
            res -= dt * Q[m + 1, j] * (fimpl[j] + fexpl[j])
        res = df.project(res, V)
        bh.apply(res.vector())
        res_norm.append(df.norm(res, 'L2'))

    uex = uexact(t0 + dt, order, V)
    print(k, max(res_norm), df.norm(df.project(uex - u[-1], V), 'L2'))