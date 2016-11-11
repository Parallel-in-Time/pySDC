from __future__ import division

import dolfin as df
import numpy as np

from pySDC_implementations.datatype_classes import fenics_mesh
from pySDC_core.Problem import ptype


class fenics_heat_unforced(ptype):
    """
    Example implementing the unforced 1D heat equation with Dirichlet-0 BC in [0,1]

    Attributes:
        V: function space
        M: mass matrix for FEM
        K: stiffness matrix incl. diffusion coefficient (and correct sign)
        bc: boundary conditions
    """

    def __init__(self, cparams, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            cparams: custom parameters for the example
            dtype_u: particle data type (will be passed parent class)
            dtype_f: acceleration data type (will be passed parent class)
        """

        # define the Dirichlet boundary
        def Boundary(x, on_boundary):
            return on_boundary

        # these parameters will be used later, so assert their existence
        assert 'c_nvars' in cparams
        assert 'nu' in cparams
        assert 'family' in cparams
        assert 'order' in cparams
        assert 'refinements' in cparams
        assert 'k' in cparams

        # add parameters as attributes for further reference
        for k,v in cparams.items():
            setattr(self,k,v)

        df.set_log_level(df.WARNING)

        df.parameters["form_compiler"]["optimize"]     = True
        df.parameters["form_compiler"]["cpp_optimize"] = True

        # set mesh and refinement (for multilevel)
        mesh = df.UnitIntervalMesh(self.c_nvars)
        # mesh = df.UnitSquareMesh(self.c_nvars[0],self.c_nvars[1])
        for i in range(self.refinements):
            mesh = df.refine(mesh)

        # define function space for future reference
        self.V = df.FunctionSpace(mesh, self.family, self.order)
        tmp = df.Function(self.V)
        print('DoFs on this level:',len(tmp.vector().array()))

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(fenics_heat_unforced,self).__init__(self.V,dtype_u,dtype_f)

        # Stiffness term (Laplace)
        u = df.TrialFunction(self.V)
        v = df.TestFunction(self.V)
        a_K = -1.0*df.inner(df.nabla_grad(u), self.nu*df.nabla_grad(v))*df.dx

        # Mass term
        a_M = u*v*df.dx

        self.M = df.assemble(a_M)
        self.K = df.assemble(a_K)

        # set boundary values
        self.bc = df.DirichletBC(self.V, df.Constant(0.0), Boundary)


    def solve_system(self,rhs,factor,u0,t):
        """
        Dolfin's linear solver for (M-dtA)u = rhs

        Args:
            rhs: right-hand side for the nonlinear system
            factor: abbrev. for the node-to-node stepsize (or any other factor required)
            u0: initial guess for the iterative solver (not used here so far)

        Returns:
            solution as mesh
        """

        A = self.M - factor*self.K
        b = fenics_mesh(rhs)
        b = self.__apply_mass_matrix(b)

        self.bc.apply(A,b.values.vector())

        u = fenics_mesh(u0)
        df.solve(A,u.values.vector(),b.values.vector())

        return u

    def eval_f(self,u,t):
        """
        Routine to evaluate both parts of the RHS

        Args:
            u: current values
            t: current time

        Returns:
            the RHS divided into two parts
        """

        f = fenics_mesh(self.V)
        f.values = df.Function(self.V,self.K*u.values.vector())
        f = self.__invert_mass_matrix(f)
        return f


    def __apply_mass_matrix(self,u):
        """
        Routine to apply mass matrix

        Args:
            u: current values

        Returns:
            M*u
        """

        me = fenics_mesh(self.V)
        me.values = df.Function(self.V,self.M*u.values.vector())

        return me


    def __invert_mass_matrix(self,u):
        """
        Helper routine to invert mass matrix

        Args:
            u: current values

        Returns:
            inv(M)*u
        """

        me = fenics_mesh(self.V)

        A = 1.0*self.M
        b = fenics_mesh(u)

        self.bc.apply(A,b.values.vector())

        df.solve(A,me.values.vector(),b.values.vector())

        return me


    def u_exact(self,t):
        """
        Routine to compute the exact solution at time t

        Args:
            t: current time

        Returns:
            exact solution
        """

        u0 = df.Expression('sin(a*x[0]) * exp(-t*nu*a*a)',a=self.k*np.pi,t=t,nu=self.nu,degree=self.order)

        me = fenics_mesh(self.V)
        me.values = df.interpolate(u0,self.V)

        return me
