from __future__ import division
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as LA

from pySDC.Problem import ptype
from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh

from examples.advection_1d_implicit.getFDMatrix import getFDMatrix



class advection_diffusion(ptype):
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

        # these parameters will be used later, so assert their existence
        assert 'nvars' in cparams
        assert 'c' in cparams
        assert 'order' in cparams
        assert 'nu' in cparams

        assert cparams['nvars']%2 == 0
        
        # add parameters as attributes for further reference
        for k,v in cparams.items():
            setattr(self,k,v)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(advection_diffusion,self).__init__(self.nvars,dtype_u,dtype_f)

        # compute dx and get discretization matrix A
        self.mesh = np.linspace(0, 1, num=self.nvars, endpoint=False)
        self.dx   = self.mesh[1] - self.mesh[0]
        self.A    = self.__get_A_laplace(self.nvars, self.nu, self.dx) \
                    + self.__get_FD(self.nvars, self.c, self.order, self.dx)

    def __get_A_laplace(self,N,nu,dx):
        """
        Helper function to assemble FD matrix A for the laplace operator in sparse format

        Args:
            N: number of dofs
            nu: diffusion coefficient
            dx: distance between two spatial nodes

        Returns:
         matrix A dense format
        """
        stencil = [1, -2, 1]
        A = sp.diags(stencil, [-1, 0, 1], shape=(N,N)).toarray()
        # periodicity
        A[0, -1] = 1.0
        A[-1, 0] = 1.0
        A *= nu / (dx**2)
        return A

    def __get_FD(self, N, c, order, dx):
        """
        :param N:
        :param c:
        :param order:
        :param d:
        :return: Matrix in sparse format
        """
        return -c*getFDMatrix(N, order, dx)

    def solve_system(self,rhs,factor,u0):
        """
        Simple linear solver for (I-dtA)u = rhs

        Args:
            rhs: right-hand side for the nonlinear system
            factor: abbrev. for the node-to-node stepsize (or any other factor required)
            u0: initial guess for the iterative solver (not used here so far)

        Returns:
            solution as mesh
        """

        me = mesh(self.nvars)
        me.values = LA.spsolve(sp.eye(self.nvars)-factor*self.A_I, rhs.values)
        return me


    def __eval_fexpl(self,u,t):
        """
        Helper routine to evaluate the explicit part of the RHS

        Args:
            u: current values (not used here)
            t: current time

        Returns:
            explicit part of RHS
        """

        fexpl        = mesh(self.nvars)
        # fexpl.values = 0.0*self.mesh
        fexpl.values = self.A_E.dot(u.values)
        return fexpl

    def __eval_fimpl(self, u,t):
        """
        Helper routine to evaluate the implicit part of the RHS

        Args:
            u: current values
            t: current time (not used here)

        Returns:
            implicit part of RHS
        """

        fimpl        = mesh(self.nvars)
        fimpl.values = self.A_I.dot(u.values)
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

        f = rhs_imex_mesh(self.nvars)
        f.impl = self.__eval_fimpl(u, t)
        f.expl = self.__eval_fexpl(u, t)
        return f


    def u_exact(self,t):
        """
        Routine to compute the exact solution at time t

        Args:
            t: current time

        Returns:
            exact solution
        """
    # code aus pfasst++
    # void exact(DVectorT& q, time t)
    # {
    # size_t n = q.size();
    # double a = 1.0 / sqrt(4 * PI * nu * (t + t0));
    # for (size_t i = 0; i < n; i++) {
    #     q[i] = 0.0;
    # }
    # for (int ii = -2; ii < 3; ii++) {
    #     for (size_t i = 0; i < n; i++) {
    #         double x = double(i) / n - 0.5 + ii - t * v;
    #         q[i] += a * exp(-x * x / (4 * nu * (t + t0)));
    #         }
    #     }
    # }

        #TODO: this has to be adjusted
        # in here for the
        me = mesh(self.nvars)
        me.values = np.cos(2.0*np.pi*(self.mesh-self.c*t))
        return me

    def get_mesh(self, form="list"):
        if form is "list":
            return [self.mesh]
        elif form is "meshgrid":
            return self.mesh
        else:
            return None

    @property
    def system_matrix(self):
        """
        Returns the system matrix
        :return:
        """
        return self.A_E + self.A_I

    @property
    def A_I(self):
        return self.__get_A_laplace(self.nvars, self.nu, self.dx)

    @property
    def A_E(self):
        return self.__get_FD(self.nvars, self.c, self.order, self.dx)


    def force_term(self, t):
        """
        This example has no force term
        :return:
        """
        return np.zeros(self.mesh.shape[0]*t.shape[0])
