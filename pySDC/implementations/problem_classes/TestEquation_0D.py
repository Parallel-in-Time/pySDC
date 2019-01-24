import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from pySDC.implementations.datatype_classes.mesh import mesh

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError


# noinspection PyUnusedLocal
class testequation0d(ptype):
    """
    Example implementing a bundle of test equations at once (via diagonal matrix)

    Attributes:
        A: digonal matrix containing the parameters
    """

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type for solution
            dtype_f: mesh data type for RHS
        """

        # these parameters will be used later, so assert their existence
        essential_keys = ['lambdas', 'u0']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        assert not any(isinstance(i, list) for i in problem_params['lambdas']), \
            'ERROR: expect flat list here, got %s' % problem_params['lambdas']
        problem_params['nvars'] = len(problem_params['lambdas'])
        assert problem_params['nvars'] > 0, 'ERROR: expect at least one lambda parameter here'

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(testequation0d, self).__init__(init=problem_params['nvars'], dtype_u=dtype_u, dtype_f=dtype_f,
                                             params=problem_params)

        self.A = self.__get_A(self.params.lambdas)

    @staticmethod
    def __get_A(lambdas):
        """
        Helper function to assemble FD matrix A in sparse format

        Args:
            lambdas (list): list of lambda parameters

        Returns:
            scipy.sparse.csc_matrix: diagonal matrix A in CSC format
        """

        A = sp.diags(lambdas)
        return A

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """

        f = self.dtype_f(self.init)
        f.values = self.A.dot(u.values)
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple linear solver for (I-factor*A)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        me = self.dtype_u(self.init)
        L = splu(sp.eye(self.params.nvars, format='csc') - factor * self.A)
        me.values = L.solve(rhs.values)
        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        me = self.dtype_u(self.init)
        me.values = self.params.u0 * np.exp(t * np.array(self.params.lambdas))
        return me
