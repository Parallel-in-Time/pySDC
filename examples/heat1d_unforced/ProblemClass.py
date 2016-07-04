from __future__ import division
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as LA

from pySDC.Problem import ptype
from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh

class heat1d_unforced(ptype):
    """
    Example implementing the unforced 1D heat equation with Dirichlet-0 BC in [0,1]

    Attributes:
        A: second-order FD discretization of the 1D laplace operator
        dx: distance between two spatial nodes
    """

    def __init__(self, cparams, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            cparams: custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        assert 'nvars' in cparams
        assert 'nu' in cparams
        assert 'k' in cparams

        assert (cparams['nvars']+1)%2 == 0

        # add parameters as attributes for further reference
        for k,v in cparams.items():
            setattr(self,k,v)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(heat1d_unforced,self).__init__(self.nvars,dtype_u,dtype_f)

        # compute dx and get discretization matrix A
        self.dx = 1/(self.nvars + 1)
        self.A = self.__get_A(self.nvars,self.nu,self.dx)


    def __get_A(self,N,nu,dx):
        """
        Helper function to assemble FD matrix A in sparse format

        Args:
            N: number of dofs
            nu: diffusion coefficient
            dx: distance between two spatial nodes

        Returns:
         matrix A in CSC format
        """

        stencil = [1, -2, 1]
        A = sp.diags(stencil,[-1,0,1],shape=(N,N))
        A *= nu / (dx**2)
        return A.tocsc()


    def solve_system(self,rhs,factor,u0,t):
        """
        Simple linear solver for (I-dtA)u = rhs

        Args:
            rhs: right-hand side for the linear system
            factor: abbrev. for the node-to-node stepsize (or any other factor required)
            u0: initial guess for the iterative solver (not used here so far)
            t: current time (e.g. for time-dependent BCs)

        Returns:
            solution as mesh
        """

        me = mesh(self.nvars)
        me.values = LA.spsolve(sp.eye(self.nvars)-factor*self.A,rhs.values)
        return me


    def eval_f(self,u,t):
        """
        Routine to evaluate the RHS

        Args:
            u: current values
            t: current time

        Returns:
            the RHS
        """

        f = mesh(self.nvars)
        f.values = self.A.dot(u.values)
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
        xvalues = np.array([(i+1)*self.dx for i in range(self.nvars)])
        me.values = np.sin(np.pi*self.k*xvalues)*np.exp(-t*self.nu*(np.pi*self.k)**2)
        return me
