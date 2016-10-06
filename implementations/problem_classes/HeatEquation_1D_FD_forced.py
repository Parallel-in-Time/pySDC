import numpy as np

from implementations.problem_classes.HeatEquation_1D_FD import heat1d

class heat1d_forced(heat1d):

    def __init__(self, problem_params, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            problem_params: custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(heat1d_forced, self).__init__(problem_params, dtype_u, dtype_f)
        pass

    def eval_f(self,u,t):
        """
        Routine to evaluate both parts of the RHS

        Args:
            u: current values
            t: current time

        Returns:
            the RHS divided into two parts
        """

        f = self.dtype_f(self.init)
        f.impl = self.__eval_fimpl(u,t)
        f.expl = self.__eval_fexpl(u,t)
        return f

    def __eval_fexpl(self,u,t):
        """
        Helper routine to evaluate the explicit part of the RHS

        Args:
            u: current values (not used here)
            t: current time

        Returns:
            explicit part of RHS
        """

        xvalues = np.array([(i+1)*self.dx for i in range(self.params.nvars)])
        fexpl = self.dtype_u(self.init)
        fexpl.values = -np.sin(np.pi*self.params.freq*xvalues)*(np.sin(t)-self.params.nu*(np.pi*self.params.freq)**2*np.cos(t))
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

        fimpl = self.dtype_u(self.init)
        fimpl.values = self.A.dot(u.values)
        return fimpl

    def u_exact(self,t):
        """
        Routine to compute the exact solution at time t

        Args:
            t: current time

        Returns:
            exact solution
        """

        me = self.dtype_u(self.init)
        xvalues = np.array([(i+1)*self.dx for i in range(self.params.nvars)])
        me.values = np.sin(np.pi*self.params.freq*xvalues)*np.cos(t)
        return me


