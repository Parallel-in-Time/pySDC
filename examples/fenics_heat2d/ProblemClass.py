from __future__ import division
import numpy as np
import dolfin as df

from pySDC.Problem import ptype
from fenics_mesh import fenics_mesh, rhs_fenics_mesh

class fenics_heat2d(ptype):
    """
    Example implementing the forced 1D heat equation with Dirichlet-0 BC in [0,1]

    Attributes:
        A: second-order FD discretization of the 1D laplace operator
        dx: distance between two spatial nodes
    """

    def __init__(self, cparams, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            cparams: custom parameters for the example
            dtype_u: particle data type (will be passed parent class)
            dtype_f: acceleration data type (will be passed parent class)
        """

        class Boundary(df.SubDomain):  # define the Dirichlet boundary
            def inside(self, x, on_boundary):
                    return on_boundary

        # these parameters will be used later, so assert their existence
        assert 'nvars' in cparams
        assert 'alpha' in cparams
        assert 'beta' in cparams
        assert 't0' in cparams

        # add parameters as attributes for further reference
        for k,v in cparams.items():
            setattr(self,k,v)

        mesh = df.UnitSquareMesh(self.nvars[0],self.nvars[1])
        self.V = df.FunctionSpace(mesh, 'Lagrange', 1)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(fenics_heat2d,self).__init__(self.V,dtype_u,dtype_f)

        # Laplace terms
        u = df.TrialFunction(self.V)
        v = df.TestFunction(self.V)
        a_K = df.inner(df.nabla_grad(u), df.nabla_grad(v))*df.dx

        # "Mass matrix" term
        a_M = u*v*df.dx

        self.M = df.assemble(a_M)
        self.K = df.assemble(a_K)

        self.u0 = df.Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',alpha=self.alpha, beta=self.beta, t=self.t0)
        self.g = df.Expression('beta - 2 - 2*alpha', beta=self.beta, alpha=self.alpha)

        boundary = Boundary()
        self.bc = df.DirichletBC(self.V, self.u0, boundary)



    def solve_system(self,rhs,factor,u0,t):
        """
        Simple linear solver for (I-dtA)u = rhs

        Args:
            rhs: right-hand side for the nonlinear system
            factor: abbrev. for the node-to-node stepsize (or any other factor required)
            u0: initial guess for the iterative solver (not used here so far)

        Returns:
            solution as mesh
        """

        A = self.M + factor*self.K

        self.u0.t = t

        self.bc.apply(A,rhs.values.vector())

        u = fenics_mesh(u0)
        df.solve(A,u.values.vector(),rhs.values.vector())

        return u


    def __eval_fexpl(self,u,t):
        """
        Helper routine to evaluate the explicit part of the RHS

        Args:
            u: current values (not used here)
            t: current time

        Returns:
            explicit part of RHS
        """

        # self.g.t = t
        fexpl = fenics_mesh(self.V)
        fexpl.values = df.interpolate(self.g,self.V)

        return fexpl

    def __eval_fimpl(self,u,t):
        """
        Helper routine to evaluate the implicit part of the RHS

        Args:
            u: current values
            t: current time (not used here)

        Returns:
            implicit part of RHS
        """

        fimpl = fenics_mesh(self.V)
        fimpl.values = df.Function(self.V,self.K*u.values.vector())

        return fimpl


    def eval_f(self,u,t):
        """
        Routine to evaluate both parts of the RHS

        Args:
            u: current values
            t: current time

        Returns:
            the RHS divided into two parts
        """

        f = rhs_fenics_mesh(u.V)
        f.impl = self.__eval_fimpl(u,t)
        f.expl = self.__eval_fexpl(u,t)
        return f


    def apply_mass_matrix(self,u):

        return self.M*u.values.vector()


    def u_exact(self,t):
        """
        Routine to compute the exact solution at time t

        Args:
            t: current time

        Returns:
            exact solution
        """
        return None

    def u_init(self,t0):

        me = fenics_mesh(self.init)
        me.values = df.interpolate(self.u0,self.V)
        return me
