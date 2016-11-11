from __future__ import division

import dolfin as df
import numpy as np

from pySDC_core.Problem import ptype
from pySDC_implementations.datatype_classes.fenics_mesh import fenics_mesh


class fenics_heat(ptype):
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
        def Boundary(x, on_boundary):
            return on_boundary

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

        df.parameters["form_compiler"]["optimize"]     = True
        df.parameters["form_compiler"]["cpp_optimize"] = True

        # set mesh and refinement (for multilevel)
        mesh = df.UnitIntervalMesh(self.c_nvars)
        # mesh = df.UnitSquareMesh(self.c_nvars[0],self.c_nvars[1])
        for i in range(self.refinements):
            mesh = df.refine(mesh)

        # self.mesh = mesh
        # define function space for future reference
        self.V = df.FunctionSpace(mesh, self.family, self.order)
        tmp = df.Function(self.V)
        print('DoFs on this level:',len(tmp.vector().array()))

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(fenics_heat,self).__init__(self.V,dtype_u,dtype_f)

        self.g = df.Expression('-sin(a*x[0]) * (sin(t) - b*a*a*cos(t))',a=np.pi,b=self.nu,t=self.t0,degree=self.order)

        # rhs in weak form
        self.w = df.Function(self.V)
        v = df.TestFunction(self.V)
        self.a_K = -self.nu*df.inner(df.nabla_grad(self.w), df.nabla_grad(v))*df.dx + self.g*v*df.dx

        # mass matrix
        u = df.TrialFunction(self.V)
        a_M = u*v*df.dx
        self.M = df.assemble(a_M)

        self.bc = df.DirichletBC(self.V, df.Constant(0.0), Boundary)


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

        sol = fenics_mesh(self.V)

        self.g.t = t
        self.w.assign(sol.values)

        v = df.TestFunction(self.V)
        F = self.w*v*df.dx - factor*self.a_K - rhs.values*v*df.dx

        du = df.TrialFunction(self.V)
        J  = df.derivative(F, self.w, du)

        problem = df.NonlinearVariationalProblem(F, self.w, self.bc, J)
        solver  = df.NonlinearVariationalSolver(problem)

        prm = solver.parameters
        prm['newton_solver']['absolute_tolerance'] = 1E-8
        prm['newton_solver']['relative_tolerance'] = 1E-7
        prm['newton_solver']['maximum_iterations'] = 25
        prm['newton_solver']['relaxation_parameter'] = 1.0

        # df.set_log_level(df.PROGRESS)

        solver.solve()

        sol.values.assign(self.w)

        return sol



    def eval_f(self,u,t):
        """
        Routine to evaluate both parts of the RHS

        Args:
            u: current values
            t: current time

        Returns:
            the RHS divided into two parts
        """

        self.g.t = t

        f = fenics_mesh(self.V)

        self.w.assign(u.values)
        f.values = df.Function(self.V,df.assemble(self.a_K))

        f = self.__invert_mass_matrix(f)

        return f



    def u_exact(self,t):
        """
        Routine to compute the exact solution at time t

        Args:
            t: current time

        Returns:
            exact solution
        """

        # u0 = df.Expression('sin(a*x[0]) * sin(a*x[1]) * cos(t)',a=np.pi,t=t,degree=self.order)
        u0 = df.Expression('sin(a*x[0]) * cos(t)',a=np.pi,t=t,degree=self.order)
        # u0 = df.Expression('sin(a*x[0]) * exp(-b*a*a*t)',a=np.pi,b=self.nu,t=t,degree=self.order)

        me = fenics_mesh(self.V)
        me.values = df.interpolate(u0,self.V)

        return me
