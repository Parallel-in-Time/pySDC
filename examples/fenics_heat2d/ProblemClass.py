from __future__ import division
import dolfin as df

import numpy as np

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

        # class Boundary(df.SubDomain):  # define the Dirichlet boundary
        #     def inside(self, x, on_boundary):
        #             return on_boundary

        def Boundary(x, on_boundary):
            return on_boundary

        # these parameters will be used later, so assert their existence
        assert 'c_nvars' in cparams
        assert 'nu' in cparams
        assert 't0' in cparams
        assert 'family' in cparams
        assert 'order' in cparams
        # assert 'levelnumber' in cparams

        # add parameters as attributes for further reference
        for k,v in cparams.items():
            setattr(self,k,v)

        df.set_log_level(df.WARNING)

        mesh = df.UnitIntervalMesh(self.c_nvars[0])#,self.c_nvars[1])
        # for i in range(self.levelnumber):
        #     mesh = df.refine(mesh)

        self.V = df.FunctionSpace(mesh, self.family, self.order)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(fenics_heat2d,self).__init__(self.V,dtype_u,dtype_f)

        # Laplace terms
        u = df.TrialFunction(self.V)
        v = df.TestFunction(self.V)
        a_K = df.inner(df.nabla_grad(u), df.nabla_grad(v))*df.dx

        # "Mass matrix" term
        a_M = u*v*df.dx

        self.M = df.assemble(a_M)
        self.K = self.nu*df.assemble(a_K)

        self.g = df.Expression('-sin(a*x[0]) * (sin(t) - b*a*a*cos(t))',a=np.pi,b=self.nu,t=self.t0,degree=self.order)
        # self.g = df.Expression('-sin(a*x[0]) * sin(a*x[1]) * (sin(t) - b*2*a*a*cos(t))',a=np.pi,b=self.nu,t=self.t0,degree=self.order)
        # self.u0 = df.Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',alpha=self.alpha, beta=self.beta, t=self.t0)
        self.u0 = df.Constant(0.0)
        self.bc = df.DirichletBC(self.V, self.u0, Boundary)



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
        b = fenics_mesh(rhs)

        self.bc.apply(A,b.values.vector())

        u = fenics_mesh(u0)
        df.solve(A,u.values.vector(),b.values.vector())

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

        self.g.t = t
        fexpl = fenics_mesh(self.V)
        fexpl.values = df.Function(self.V,self.M*df.interpolate(self.g,self.V).vector())

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
        fimpl.values = df.Function(self.V,-1.0*self.K*u.values.vector())

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

        me = fenics_mesh(self.V)
        me.values = df.Function(self.V,self.M*u.values.vector())

        return me


    def invert_mass_matrix(self,u):

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

        # u0 = df.Expression('sin(a*x[0]) * sin(a*x[1]) * cos(t)',a=np.pi,t=t,degree=self.order)
        u0 = df.Expression('sin(a*x[0]) * cos(t)',a=np.pi,t=t)

        me = fenics_mesh(self.init)
        me.values = df.interpolate(u0,self.V)

        return me
