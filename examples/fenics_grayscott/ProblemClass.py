from __future__ import division
import dolfin as df

import numpy as np
import random

from pySDC.Problem import ptype
from pySDC.datatype_classes.fenics_mesh import fenics_mesh,rhs_fenics_mesh

class fenics_grayscott(ptype):
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
        # mesh = df.UnitIntervalMesh(self.c_nvars)
        # mesh = df.UnitSquareMesh(self.c_nvars[0],self.c_nvars[1])
        mesh = df.IntervalMesh(self.c_nvars,0,100)
        # mesh = df.RectangleMesh(0.0,0.0,2.0,2.0,self.c_nvars[0],self.c_nvars[1])
        for i in range(self.refinements):
            mesh = df.refine(mesh)

        # self.mesh = mesh
        # define function space for future reference
        V = df.FunctionSpace(mesh, self.family, self.order)
        self.V = V*V

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(fenics_grayscott,self).__init__(self.V,dtype_u,dtype_f)

        # rhs in weak form
        self.w = df.Function(self.V)
        q1,q2 = df.TestFunctions(self.V)

        self.w1,self.w2 = df.split(self.w)

        self.F1 = (-self.Du*df.inner(df.nabla_grad(self.w1), df.nabla_grad(q1)) - self.w1*(self.w2**2)*q1 + self.A*(1-self.w1)*q1)*df.dx
        self.F2 = (-self.Dv*df.inner(df.nabla_grad(self.w2), df.nabla_grad(q2)) + self.w1*(self.w2**2)*q2 - self.B*    self.w2*q2)*df.dx
        self.F = self.F1+self.F2

        # mass matrix
        u1,u2 = df.TrialFunctions(self.V)
        a_M = u1*q1*df.dx
        M1 = df.assemble(a_M)
        a_M = u2*q2*df.dx
        M2 = df.assemble(a_M)
        self.M = M1+M2

        # self.bc = df.DirichletBC(self.V, df.Constant(0.0), Boundary)

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

        # self.bc.apply(A,b.values.vector())

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

        # self.g.t = t
        self.w.assign(sol.values)

        q1,q2 = df.TestFunctions(self.V)
        w1,w2 = df.split(self.w)
        r1,r2 = df.split(rhs.values)
        F1 = w1*q1*df.dx - factor*self.F1 - r1*q1*df.dx
        F2 = w2*q2*df.dx - factor*self.F2 - r2*q2*df.dx
        F = F1+F2
        du = df.TrialFunction(self.V)
        J  = df.derivative(F, self.w, du)

        problem = df.NonlinearVariationalProblem(F, self.w, [], J)
        solver  = df.NonlinearVariationalSolver(problem)

        prm = solver.parameters
        prm['newton_solver']['absolute_tolerance'] = 1E-09
        prm['newton_solver']['relative_tolerance'] = 1E-08
        prm['newton_solver']['maximum_iterations'] = 100
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

        f = fenics_mesh(self.V)

        self.w.assign(u.values)
        f.values = df.Function(self.V,df.assemble(self.F))

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

        class InitialConditions(df.Expression):
            def __init__(self):
                random.seed(2)
                pass
            def eval(self, values, x):
                # if df.between(x[0],(0.375,0.625)):
                #   # values[1] = 0.25*np.power(np.sin(8*np.pi*x[0]),2)
                #   # values[0] = 1 - 2*values[1]
                #   values[0] = 0.5 + random.random()
                #   values[1] = 0.25
                # else:
                #   values[1] = 0
                #   values[0] = 1
                values[0] = 1 - 0.5*np.power(np.sin(np.pi*x[0]/100),100)
                values[1] = 0.25*np.power(np.sin(np.pi*x[0]/100),100)
            def value_shape(self):
                return (2,)

        # class InitialConditions(df.Expression):
        #   def eval(self, values, x):
        #     if df.between(x[0],(0.75,1.25)) and df.between(x[1],(0.75,1.25)):
        #       values[1] = 0.25*np.power(np.sin(4*np.pi*x[0]),2)*np.power(np.sin(4*np.pi*x[1]),2)
        #       values[0] = 1 - 2*values[1]
        #     else:
        #       values[1] = 0
        #       values[0] = 1
        #   def value_shape(self):
        #     return (2,)


        # class InitialConditions(df.Expression):
        #    def __init(self):
        #        random.seed(2)
        #
        #    def eval(self, values, x):
        #      if df.between(x[0],(0.475,0.525)) and df.between(x[1],(0.475,0.525)):
        #          # values[1] = 0.25 + random.uniform(-0.01,0.01)
        #          # values[0] = 0.5 + random.uniform(-0.01,0.01)
        #          values[1] = 1
        #          values[0] = 0
        #      else:
        #          values[1] = 0
        #          values[0] = 1
        #    def value_shape(self):
        #      return (2,)

        uinit = InitialConditions()


        me = fenics_mesh(self.V)
        me.values = df.interpolate(uinit,self.V)
        # u1,u2 = df.split(me.values)
        # df.plot(u1,interactive = True)
        # exit()
        return me
