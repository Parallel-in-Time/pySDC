import time

import cffi
import dolfinx.fem.petsc as petsc
import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from ufl import div, ds, dx, grad, inner


class LocalAssembler():
    def __init__(self, form):
        self.form = fem.form(form)
        self.update_coefficients()
        self.update_constants()
        cell_form_pos = self.form.ufcx_form.form_integral_offsets[0]
        subdomain_ids = self.form._cpp_object.integral_ids(fem.IntegralType.cell)

        assert(len(subdomain_ids) == 1)
        assert(subdomain_ids[0] == -1)
       
        is_complex = np.issubdtype(ScalarType, np.complexfloating)
        nptype = "complex128" if is_complex else "float64"       
        self.kernel = getattr(self.form.ufcx_form.form_integrals[cell_form_pos],f"tabulate_tensor_{nptype}")
        self.active_cells = self.form._cpp_object.domains(fem.IntegralType.cell,subdomain_ids[0])
        assert len(self.form.function_spaces) == self.form.rank
        self.local_shape = [0 for _ in range(self.form.rank)]
        for i, V in enumerate(self.form.function_spaces):
            self.local_shape[i] = V.dofmap.dof_layout.block_size * V.dofmap.dof_layout.num_dofs

        e0 = self.form.function_spaces[0].element
        if self.form.rank == 1:
            e1 = self.form.function_spaces[0].element
        elif self.form.rank == 2:
            e1 = self.form.function_spaces[1].element
        else:
            raise RuntimeError(f"Assembly for tensor of tensor of {rank=} is not supported")

        needs_transformation_data = e0.needs_dof_transformations or e1.needs_dof_transformations or \
            self.form._cpp_object.needs_facet_permutations
        if needs_transformation_data:
            raise NotImplementedError("Dof transformations not implemented")

        self.ffi = cffi.FFI()
        V = self.form.function_spaces[0]
        self.x_dofs = V.mesh.geometry.dofmap
        self.local_tensor = np.zeros(self.local_shape, dtype=ScalarType)

    def update_coefficients(self):
        self.coeffs = fem.assemble.pack_coefficients(self.form._cpp_object)[(fem.IntegralType.cell, -1)]

    def update_constants(self):
        self.consts = fem.assemble.pack_constants(self.form._cpp_object)

    def update(self):
        self.update_coefficients()
        self.update_constants()

    def assemble(self, i:int):
        x = self.form.function_spaces[0].mesh.geometry.x
        x_dofs = self.x_dofs[i]
        geometry = np.zeros((len(x_dofs), 3), dtype=np.float64)
        geometry[:, :] = x[x_dofs]


        facet_index = np.zeros(0, dtype=np.intc)
        facet_perm = np.zeros(0, dtype=np.uint8)
        if self.coeffs.shape == (0, 0):
            coeffs = np.zeros(0, dtype=ScalarType)
        else:
            coeffs = self.coeffs[i,:]
        ffi_fb = self.ffi.from_buffer
        self.local_tensor.fill(0)
        self.kernel(ffi_fb(self.local_tensor), ffi_fb(coeffs), ffi_fb(self.consts), ffi_fb(geometry),
               ffi_fb(facet_index), ffi_fb(facet_perm))
        return self.local_tensor


N = 20
msh = mesh.create_box(MPI.COMM_WORLD, [np.array([0.0, 0.0, 0.0]),
                                  np.array([2.0, 1.0, 1.0])], [N, N, N],
                 mesh.CellType.tetrahedron)

Q = fem.FunctionSpace(msh, ("DG", 2))

p = ufl.TrialFunction(Q)
q = ufl.TestFunction(Q)
a = inner(p, q) *dx
lhs_assembler = LocalAssembler(a)

W = fem.VectorFunctionSpace(msh,  ("Lagrange", 1))
uh = fem.Function(W)
uh.interpolate(lambda x: (x[0], 2*x[1], x[2]**2))

l = inner(div(uh), q)*dx
rhs_assembler = LocalAssembler(l)

w1 = fem.Function(Q)
bs = Q.dofmap.bs
start = time.perf_counter()
for cell in range(msh.topology.index_map(msh.topology.dim).size_local):
    A = lhs_assembler.assemble(cell)
    b = rhs_assembler.assemble(cell)
    cell_dofs = Q.dofmap.cell_dofs(cell)
    unrolled_dofs = np.zeros(len(cell_dofs)*bs, dtype=np.int32)
    for i, dof in enumerate(cell_dofs):
        for j in range(bs):
            unrolled_dofs[i*bs+j] = dof*bs+j
    w1.x.array[unrolled_dofs] = np.linalg.solve(A, b)
w1.x.scatter_forward()
end = time.perf_counter()
print(f"Local assembly {end-start:.2e}")


start = time.perf_counter()
w2 = fem.Function(Q)
problem = petsc.LinearProblem(a, l, u=w2)
problem.solve()
w2.x.scatter_forward()
end = time.perf_counter()
print(f"Global assembly {end-start:.2e}")


from dolfinx.io import VTXWriter

with VTXWriter(msh.comm, "w1.bp", [w1]) as vtx:
    vtx.write(0)

with VTXWriter(msh.comm, "w2.bp", [w2]) as vtx:
    vtx.write(0)

assert np.allclose(w1.x.array, w2.x.array)