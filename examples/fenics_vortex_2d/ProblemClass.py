from __future__ import division
import dolfin as df

import numpy as np

from pySDC.Problem import ptype
from pySDC.datatype_classes.fenics_mesh import fenics_mesh,rhs_fenics_mesh

import matplotlib.pyplot as plt

class fenics_vortex_2d(ptype):
    """
    Example implementing the forced 1D heat equation with Dirichlet-0 BC in [0,1]

    Attributes:
        V: function space
        M: mass matrix for FEM
        K: stiffness matrix incl. diffusion coefficient (and correct sign)
        g: forcing term
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
        # def Boundary(x, on_boundary):
        #     return on_boundary

        # Sub domain for Periodic boundary condition
        class PeriodicBoundary(df.SubDomain):

            # Left boundary is "target domain" G
            def inside(self, x, on_boundary):
                # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
                return bool((df.near(x[0], 0) or df.near(x[1], 0)) and
                        (not ((df.near(x[0], 0) and df.near(x[1], 1)) or
                                (df.near(x[0], 1) and df.near(x[1], 0)))) and on_boundary)

            def map(self, x, y):
                if df.near(x[0], 1) and df.near(x[1], 1):
                    y[0] = x[0] - 1.
                    y[1] = x[1] - 1.
                elif df.near(x[0], 1):
                    y[0] = x[0] - 1.
                    y[1] = x[1]
                else:   # near(x[1], 1)
                    y[0] = x[0]
                    y[1] = x[1] - 1.

        # these parameters will be used later, so assert their existence
        assert 'c_nvars' in cparams
        assert 'nu' in cparams
        assert 't0' in cparams
        assert 'family' in cparams
        assert 'order' in cparams
        assert 'refinements' in cparams

        # add parameters as attributes for further reference
        for k,v in cparams.items():
            setattr(self,k,v)

        df.set_log_level(df.WARNING)

        # set mesh and refinement (for multilevel)
        # mesh = df.UnitIntervalMesh(self.c_nvars)
        mesh = df.UnitSquareMesh(self.c_nvars[0],self.c_nvars[1])
        for i in range(self.refinements):
            mesh = df.refine(mesh)

        self.mesh = df.Mesh(mesh)

        self.bc = PeriodicBoundary()

        # define function space for future reference
        self.V = df.FunctionSpace(mesh, self.family, self.order, constrained_domain=self.bc)
        tmp = df.Function(self.V)
        print('DoFs on this level:',len(tmp.vector().array()))

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(fenics_vortex_2d,self).__init__(self.V,dtype_u,dtype_f)

        w = df.TrialFunction(self.V)
        v = df.TestFunction(self.V)

        # Stiffness term (diffusion)
        a_K = df.inner(df.nabla_grad(w), df.nabla_grad(v))*df.dx

        # Mass term
        a_M = w*v*df.dx

        self.M = df.assemble(a_M)
        self.K = df.assemble(a_K)

    def solve_system(self,rhs,factor,u0,t):
        """
        Dolfin's linear solver for (M-dtA)u = rhs

        Args:
            rhs: right-hand side for the nonlinear system
            factor: abbrev. for the node-to-node stepsize (or any other factor required)
            u0: initial guess for the iterative solver (not used here so far)
            t: current time (e.g. for time-dependent BCs)

        Returns:
            solution as mesh
        """

        A = self.M + self.nu*factor*self.K
        b = fenics_mesh(rhs)

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


        A = 1.0*self.K
        b = self.apply_mass_matrix(u)
        psi = fenics_mesh(self.V)
        df.solve(A,psi.values.vector(),b.values.vector())

        fexpl = fenics_mesh(self.V)
        fexpl.values = df.project(df.Dx(psi.values,1)*df.Dx(u.values,0) - df.Dx(psi.values,0)*df.Dx(u.values,1),self.V)

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

        tmp = fenics_mesh(self.V)
        tmp.values = df.Function(self.V,-1.0*self.nu*self.K*u.values.vector())
        fimpl = self.__invert_mass_matrix(tmp)

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

        f = rhs_fenics_mesh(self.V)
        f.impl = self.__eval_fimpl(u,t)
        f.expl = self.__eval_fexpl(u,t)
        return f


    def apply_mass_matrix(self,u):
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

        w = df.Expression('r*(1-pow(tanh(r*((0.75-4) - x[1])),2)) + r*(1-pow(tanh(r*(x[1] - (0.25-4))),2)) - \
                           r*(1-pow(tanh(r*((0.75-3) - x[1])),2)) + r*(1-pow(tanh(r*(x[1] - (0.25-3))),2)) - \
                           r*(1-pow(tanh(r*((0.75-2) - x[1])),2)) + r*(1-pow(tanh(r*(x[1] - (0.25-2))),2)) - \
                           r*(1-pow(tanh(r*((0.75-1) - x[1])),2)) + r*(1-pow(tanh(r*(x[1] - (0.25-1))),2)) - \
                           r*(1-pow(tanh(r*((0.75-0) - x[1])),2)) + r*(1-pow(tanh(r*(x[1] - (0.25-0))),2)) - \
                           r*(1-pow(tanh(r*((0.75+1) - x[1])),2)) + r*(1-pow(tanh(r*(x[1] - (0.25+1))),2)) - \
                           r*(1-pow(tanh(r*((0.75+2) - x[1])),2)) + r*(1-pow(tanh(r*(x[1] - (0.25+2))),2)) - \
                           r*(1-pow(tanh(r*((0.75+3) - x[1])),2)) + r*(1-pow(tanh(r*(x[1] - (0.25+3))),2)) - \
                           r*(1-pow(tanh(r*((0.75+4) - x[1])),2)) + r*(1-pow(tanh(r*(x[1] - (0.25+4))),2)) - \
                           d*2*a*cos(2*a*(x[0]+0.25))',d=self.delta,r=self.rho,a=np.pi,degree=self.order)

        me = fenics_mesh(self.V)
        me.values = df.interpolate(w,self.V)

        # df.plot(me.values)
        # df.interactive()
        # exit()

        return me
