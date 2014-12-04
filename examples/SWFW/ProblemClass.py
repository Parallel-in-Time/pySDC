import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as LA

from pySDC.Problem import ptype
from pySDC.datatype_classes.complex_mesh import mesh, rhs_imex_mesh

class swfw_scalar(ptype):
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
        assert 'lambda_s' in cparams
        assert 'lambda_f' in cparams
        assert 'u0' in cparams

        # add parameters as attributes for further reference
        for k,v in cparams.items():
            setattr(self,k,v)

        self.nvars = (self.lambda_s.size,self.lambda_f.size)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(swfw_scalar,self).__init__(self.nvars,dtype_u,dtype_f)



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
        for i in range(self.lambda_s.size):
            for j in range(self.lambda_f.size):
               me.values[i,j] = rhs.values[i,j]/(1-factor*self.lambda_f[j])

        return me


    def __eval_fexpl(self,u,t):
        """
        Helper routine to evaluate the explicit part of the RHS

        Args:
            u: current values
            t: current time (not used here)

        Returns:
            explicit part of RHS
        """

        fexpl = mesh(self.nvars)
        for i in range(self.lambda_s.size):
            for j in range(self.lambda_f.size):
                fexpl.values[i,j] = self.lambda_s[i]*u.values[i,j]
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

        fimpl = mesh(self.nvars)
        for i in range(self.lambda_s.size):
            for j in range(self.lambda_f.size):
                fimpl.values[i,j] = self.lambda_f[j]*u.values[i,j]

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
        f.impl = self.__eval_fimpl(u,t)
        f.expl = self.__eval_fexpl(u,t)
        return f


    def u_exact(self,t):
        """
        Routine to compute the exact solution at time t

        Args:
            t: current time

        Returns:
            exact solution
        """

        me = mesh(self.nvars)
        for i in range(self.lambda_s.size):
            for j in range(self.lambda_f.size):
                me.values[i,j] = self.u0*np.exp((self.lambda_f[j]+self.lambda_s[i])*t)
        return me
