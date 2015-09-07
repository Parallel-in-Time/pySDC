from __future__ import division
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as LA

from pySDC.Problem import ptype
from pySDC.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.tools.transfer_tools import to_sparse

class test_eq(ptype):
    """
    Example implementing the test_equation, useful for experimental computations of the stability function

    Attributes:
        lambda: yes the famous one...
    """

    def __init__(self, cparams, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            cparams: custom parameters for the example
            dtype_u: temperature on a mesh (will be passed parent class)
            dtype_f: temperature per time unit on a mesh (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        assert 'lamb' in cparams

        # add parameters as attributes for further reference
        for k,v in cparams.items():
            setattr(self,k,v)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(test_eq,self).__init__(self.nvars,dtype_u,dtype_f)

        # to sustain a flawless work with the matrix classes of LinearPFASST we define a Matrix A of the shape (1,1)
        self.A = np.eye(1)*self.lamb
        self.nvars = 1

    def solve_system(self,rhs,factor,u0,t):
        """
        Simple linear solver for (I-dt*lambda)u = rhs

        Args:
            rhs: right-hand side for the nonlinear system
            factor: abbrev. for the node-to-node stepsize (or any other factor required)
            u0: initial guess for the iterative solver (not used here so far)
            t: current time (e.g. for time-dependent BCs)

        Returns:
            solution as mesh
        """

        me = mesh(self.nvars)
        me.values = np.ones(1)/(1-self.lamb*factor)*rhs.values
        # me.values = LA.spsolve(sp.eye(self.nvars)-factor*self.A,rhs.values)
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

        fexpl = mesh(self.nvars)
        # fexpl.values = -np.sin(np.pi*xvalues)*(np.sin(t)-self.nu*np.pi**2*np.cos(t))
        fexpl.values = np.zeros(self.nvars)
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
        fimpl.values = self.lamb * u.values
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
        me = mesh(1)
        me.values[0] = np.exp(self.lamb*t)
        return me


    def get_mesh(self, form="list"):
        """
        Returns the mesh the problem is computed on.

        :param form: the form in which the mesh is needed
        :return: depends on form
        """

        if form is "list":
            return [np.zeros(1)]
        elif form is "meshgrid":
            return np.zeros(1)
        else:
            return None


    @property
    def system_matrix(self):
        """
        Returns the system matrix
        :return:
        """
        return self.A

    @property
    def A_I(self):
        return self.A

    @property
    def A_E(self):
        return np.zeros(self.A.shape)

    def force_term(self, t):
        """
        For the linear matrix framework it is possible to
        deal with forcing terms as long they only depend on t.
        :param t: time point , array
        :return: forcing term of
        """
        if type(t) is np.ndarray:
            return np.zeros(self.xvalues.shape[0]*t.shape[0])
            # return np.hstack(map(lambda tau: -np.sin(np.pi*self.xvalues)*(np.sin(tau)-self.nu*np.pi**2*np.cos(tau)), t))
        else:
            # return -np.sin(np.pi*self.xvalues)*(np.sin(t)-self.nu*np.pi**2*np.cos(t))
            return np.zeros(self.nvars)


