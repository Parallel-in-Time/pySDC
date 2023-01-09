import logging

import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx as dfx
import ufl
from matplotlib import pyplot as plt

import numpy as np

from pySDC.core.Errors import ParameterError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh

# noinspection PyUnusedLocal
class fenicsx_heat(ptype):
    """
    Example implementing the forced 1D heat equation with Dirichlet-0 BC in [0,1]

    Attributes:
        V: function space
        M: mass matrix for FEM
        K: stiffness matrix incl. diffusion coefficient (and correct sign)
        g: forcing term
        bc: boundary conditions
    """

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=imex_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: FEniCS mesh data type (will be passed to parent class)
            dtype_f: FEniCS mesh data data type with implicit and explicit parts (will be passed to parent class)
        """

        # define the Dirichlet boundary
        # def Boundary(x, on_boundary):
        #     return on_boundary

        if 'comm' not in problem_params:
            problem_params['comm'] = MPI.COMM_WORLD

        # these parameters will be used later, so assert their existence
        essential_keys = ['nelems', 't0', 'family', 'order', 'refinements', 'nu', 'comm']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # Define mesh
        domain = dfx.mesh.create_interval(problem_params['comm'], nx=problem_params['nelems'], points=np.array([0, 1]))
        self.V = dfx.fem.FunctionSpace(domain, (problem_params['family'], problem_params['order']))
        self.x = ufl.SpatialCoordinate(domain)
        tmp = dfx.fem.Function(self.V)
        nx = len(tmp.x.array)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(fenicsx_heat, self).__init__((nx, problem_params['comm'], np.dtype('float64')), dtype_u, dtype_f, problem_params)

        # Create boundary condition
        fdim = domain.topology.dim - 1
        boundary_facets = dfx.mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
        self.bc = dfx.fem.dirichletbc(PETSc.ScalarType(0), dfx.fem.locate_dofs_topological(self.V, fdim, boundary_facets), self.V)

        # Stiffness term (Laplace) and mass term
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        a_K = -1.0 * ufl.dot(ufl.grad(u), self.params.nu * ufl.grad(v)) * ufl.dx
        a_M = u * v * ufl.dx

        self.K = dfx.fem.petsc.assemble_matrix(dfx.fem.form(a_K), bcs=[self.bc])
        self.K.assemble()

        self.M = dfx.fem.petsc.assemble_matrix(dfx.fem.form(a_M), bcs=[self.bc])
        self.M.assemble()

        # set forcing term
        self.g = dfx.fem.Function(self.V)
        t = self.params.t0
        self.g.interpolate(lambda x: -np.sin(2 * np.pi*x[0]) * (np.sin(t) - 4 * self.params.nu*np.pi*np.pi*np.cos(t)))

        self.tmp_u = dfx.fem.Function(self.V)
        self.tmp_f = dfx.fem.Function(self.V)

        self.solver = PETSc.KSP().create(domain.comm)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)

    @staticmethod
    def convert_to_fenicsx_vector(input, output):
        output.x.array[:] = input[:]

    @staticmethod
    def convert_from_fenicsx_vector(input, output):
        output[:] = input.x.array[:]

    def solve_system(self, rhs, factor, u0, t):
        """
        Dolfin's linear solver for (M-dtA)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the nonlinear system
            factor (float): abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u_: initial guess for the iterative solver (not used here so far)
            t (float): current time

        Returns:
            dtype_u: solution as mesh
        """

        self.convert_to_fenicsx_vector(input=u0, output=self.tmp_u)
        self.convert_to_fenicsx_vector(input=rhs, output=self.tmp_f)
        b = dfx.fem.Function(self.V)
        self.M.mult(self.tmp_f.vector, b.vector)

        dfx.fem.petsc.set_bc(b.vector, [self.bc])
        self.solver.setOperators(self.M - factor * self.K)
        self.solver.solve(b.vector, self.tmp_u.vector)
        # tmp_u.x.scatter_forward()

        u = self.dtype_u(self.init)
        self.convert_from_fenicsx_vector(input=self.tmp_u, output=u)

        return u

    def apply_mass_matrix(self, u):

        self.convert_to_fenicsx_vector(input=u, output=self.tmp_u)
        self.M.mult(self.tmp_u.vector, self.tmp_f.vector)
        uM = self.dtype_u(self.init)
        self.convert_from_fenicsx_vector(input=self.tmp_f, output=uM)
        return uM

    def eval_f(self, u, t):
        """
        Routine to evaluate both parts of the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS divided into two parts
        """

        f = self.dtype_f(self.init)
        b = dfx.fem.Function(self.V)

        self.convert_to_fenicsx_vector(input=u, output=self.tmp_u)
        self.K.mult(self.tmp_u.vector, b.vector)


        self.solver.setOperators(self.M)
        self.solver.solve(b.vector, self.tmp_f.vector)
        self.convert_from_fenicsx_vector(input=self.tmp_f, output=f.impl)

        self.g.interpolate(lambda x: -np.sin(2 * np.pi * x[0]) * (np.sin(t) - self.params.nu * np.pi * np.pi * 4 * np.cos(t)))
        # self.M.mult(self.g.vector, self.tmp_f.vector)
        # self.convert_from_fenicsx_vector(input=self.tmp_f, output=f.expl)
        self.convert_from_fenicsx_vector(input=self.g, output=f.expl)

        return f

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        u0 = dfx.fem.Function(self.V)
        u0.interpolate(lambda x: np.sin(2 * np.pi * x[0]) * np.cos(t))

        me = self.dtype_u(self.init)
        self.convert_from_fenicsx_vector(input=u0, output=me)

        return me


#noinspection PyUnusedLocal
class fenicsx_heat_mass(fenicsx_heat):
    """
    Example implementing the forced 1D heat equation with Dirichlet-0 BC in [0,1], expects mass matrix sweeper

    """

    def solve_system(self, rhs, factor, u0, t):
        """
        Dolfin's linear solver for (M-dtA)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the nonlinear system
            factor (float): abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u_: initial guess for the iterative solver (not used here so far)
            t (float): current time

        Returns:
            dtype_u: solution as mesh
        """

        self.convert_to_fenicsx_vector(input=u0, output=self.tmp_u)
        self.convert_to_fenicsx_vector(input=rhs, output=self.tmp_f)

        dfx.fem.petsc.set_bc(self.tmp_f.vector, [self.bc])
        self.solver.setOperators(self.M - factor * self.K)
        self.solver.solve(self.tmp_f.vector, self.tmp_u.vector)
        # tmp_u.x.scatter_forward()

        u = self.dtype_u(self.init)
        self.convert_from_fenicsx_vector(input=self.tmp_u, output=u)

        return u

    def eval_f(self, u, t):
        """
        Routine to evaluate both parts of the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS divided into two parts
        """

        f = self.dtype_f(self.init)

        self.convert_to_fenicsx_vector(input=u, output=self.tmp_u)
        self.K.mult(self.tmp_u.vector, self.tmp_f.vector)
        self.convert_from_fenicsx_vector(input=self.tmp_f, output=f.impl)

        self.g.interpolate(lambda x: -np.sin(2 * np.pi * x[0]) * (np.sin(t) - self.params.nu * np.pi * np.pi * 4 * np.cos(t)))
        self.M.mult(self.g.vector, self.tmp_f.vector)
        self.convert_from_fenicsx_vector(input=self.tmp_f, output=f.expl)

        return f
