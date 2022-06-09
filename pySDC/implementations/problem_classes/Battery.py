import numpy as np

from pySDC.core.Errors import ParameterError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class battery(ptype):
    """
    Example implementing the battery drain model as in the description in the PinTSimE project
    Attributes:
        A: system matrix, representing the 2 ODEs
    """

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=imex_mesh):
        """
        Initialization routine
        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type for solution
            dtype_f: mesh data type for RHS
        """

        problem_params['nvars'] = 2

        # these parameters will be used later, so assert their existence
        essential_keys = ['Vs', 'Rs', 'C', 'R', 'L', 'alpha', 'V_ref']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(battery, self).__init__(init=(problem_params['nvars'], None, np.dtype('float64')),
                                      dtype_u=dtype_u, dtype_f=dtype_f, params=problem_params)

        self.A = np.zeros((2, 2))

    def switch_estimator(self, u):
        """
            Method to estimate a discrete event (switch)
        """

        L = S.levels[0]

        t_interp = [L.time + L.dt * L.sweep.coll.nodes[m] for m in range(len(L.sweep.coll.nodes))]
            
        vC = []
        for m in range(1, len(u)):
            vC.append(u[m])

        p = scipy.interpolate.interp1d(t_interp, vC, 'cubic', bounds_error=False)

        def switch_examiner(x):
            """
                Routine to define root problem
            """

            return L.prob.params.V_ref - p(x)
            
        t_switch = scipy.optimize.fsolve(switch_examiner, t_interp[2])
            
        # next subinterval
        t_next = [t_interp[-1] + L.dt * L.sweep.coll.nodes[m] for m in range(len(L.sweep.coll.nodes))]

        # Looking for the event
        flag_event = []    
        for m in range(len(t_next)-1):
            if t_next[m] <= t_switch <= t_next[m+1]:
                flag_event.append(True)
            
            else:
                flag_event.append(False)
                    
        print(t_switch)
        print(t_interp)
        print(t_next)
        return t_switch, flag_event

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS
        Args:
            u (dtype_u): current values
            t (float): current time
        Returns:
            dtype_f: the RHS
        """

        f = self.dtype_f(self.init, val=0.0)
        f.impl[:] = self.A.dot(u)

        if u[1] <= self.params.V_ref:
            f.expl[0] = self.params.Vs / self.params.L

        else:
            f.expl[0] = 0

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

        self.A = np.zeros((2, 2))
        
        #t_switch, flag_event = switch_estimator(self, rhs[1])

        if rhs[1] <= self.params.V_ref:
            self.A[0, 0] = -(self.params.Rs + self.params.R) / self.params.L
            #print("Below reference!")

        else:
            self.A[1, 1] = -1 / (self.params.C * self.params.R)
            #print("Over reference!")

        me = self.dtype_u(self.init)
        me[:] = np.linalg.solve(np.eye(self.params.nvars) - factor * self.A, rhs)
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

        me[0] = 0.0  # cL
        me[1] = self.params.alpha * self.params.V_ref  # vC

        return me
