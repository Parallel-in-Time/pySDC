import numpy as np
import logging

from pySDC.core.Sweeper import sweeper, _Pars
from pySDC.core.Errors import ParameterError
from pySDC.implementations.datatype_classes.particles import particles, fields, acceleration
from pySDC.implementations.sweeper_classes.Runge_Kutta import ButcherTableau
from copy import deepcopy
from pySDC.implementations.sweeper_classes.Runge_Kutta import RungeKutta


class RungeKuttaNystrom(RungeKutta):
    """
    Runge-Kutta scheme that fits the interface of a sweeper.
    Actually, the sweeper idea fits the Runge-Kutta idea when using only lower triangular rules, where solutions
    at the nodes are successively computed from earlier nodes. However, we only perform a single iteration of this.

    We have two choices to realise a Runge-Kutta sweeper: We can choose Q = Q_Delta = <Butcher tableau>, but in this
    implementation, that would lead to a lot of wasted FLOPS from integrating with Q and then with Q_Delta and
    subtracting the two. For that reason, we built this new sweeper, which does not have a preconditioner.

    This class only supports lower triangular Butcher tableaux such that the system can be solved with forward
    substitution. In this way, we don't get the maximum order that we could for the number of stages, but computing the
    stages is much cheaper. In particular, if the Butcher tableaux is strictly lower triangular, we get an explicit
    method, which does not require us to solve a system of equations to compute the stages.

    Please be aware that all fundamental parameters of the Sweeper are ignored. These include

     - num_nodes
     - collocation_class
     - initial_guess
     - QI

    All of these variables are either determined by the RK rule, or are not part of an RK scheme.

    Attribues:
        butcher_tableau (ButcherTableau): Butcher tableau for the Runge-Kutta scheme that you want
    """

    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """
        # set up logger
        self.logger = logging.getLogger('sweeper')

        essential_keys = ['butcher_tableau']
        for key in essential_keys:
            if key not in params:
                msg = 'need %s to instantiate step, only got %s' % (key, str(params.keys()))
                self.logger.error(msg)
                raise ParameterError(msg)

        # check if some parameters are set which only apply to actual sweepers
        for key in ['initial_guess', 'collocation_class', 'num_nodes']:
            if key in params:
                self.logger.warning(f'"{key}" will be ignored by Runge-Kutta sweeper')

        # set parameters to their actual values
        params['initial_guess'] = 'zero'
        params['collocation_class'] = type(params['butcher_tableau'])
        params['num_nodes'] = params['butcher_tableau'].num_nodes

        # disable residual computation by default
        params['skip_residual_computation'] = params.get(
            'skip_residual_computation', ('IT_CHECK', 'IT_FINE', 'IT_COARSE', 'IT_UP', 'IT_DOWN')
        )

        self.params = _Pars(params)

        self.coll = params['butcher_tableau']
        self.coll_bar = params['butcher_tableau_bar']

        # This will be set as soon as the sweeper is instantiated at the level
        self.__level = None

        self.parallelizable = False
        self.QI = self.coll.Qmat
        self.Qx = self.coll_bar.Qmat

    @classmethod
    def get_update_order(cls):
        """
        Get the order of the lower order method for doing adaptivity. Only applies to embedded methods.
        """
        raise NotImplementedError(
            f"There is not an update order for RK scheme \"{cls.__name__}\" implemented. Maybe it is not an embedded scheme?"
        )

    def get_full_f(self, f):
        """
        Get the full right hand side as a `mesh` from the right hand side

        Args:
            f (dtype_f): Right hand side at a single node

        Returns:
            mesh: Full right hand side as a mesh
        """

        if type(f) in [particles, fields, acceleration]:
            return f
        else:
            raise NotImplementedError(f'Type \"{type(f)}\" not implemented in Runge-Kutta sweeper')

    def integrate(self):
        """
        Integrates the right-hand side

        Returns:
            list of dtype_u: containing the integral as values
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        p = []

        # integrate RHS over all collocation nodes
        for m in range(1, self.coll.num_nodes + 1):
            # new instance of dtype_u, initialize values with 0
            p.append(P.dtype_u(P.init, val=0.0))

        return p

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes

        Returns:
            None
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked
        assert L.status.sweep <= 1, "RK schemes are direct solvers. Please perform only 1 iteration!"

        # get number of collocation nodes for easier access
        M = self.coll.num_nodes

        for m in range(0, M):
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            rhs = deepcopy(L.u[0])
            rhs.pos += L.dt * self.coll.nodes[m + 1] * L.u[0].vel

            for j in range(1, m + 1):
                # build RHS from f-terms (containing the E field) and the B field

                f = P.build_f(L.f[j], L.u[j], L.time + L.dt * self.coll.nodes[j])
                rhs.pos += L.dt**2 * self.Qx[m + 1, j] * self.get_full_f(f)
                """

                Implicit part only works for Velocity-Verlet scheme
                Boris solver for the implicit part

                """

                if self.coll.implicit:
                    ck = rhs.vel * 0.0
                    L.f[3] = P.eval_f(rhs, L.time + L.dt)
                    rhs.vel = P.boris_solver(ck, L.dt, L.f[0], L.f[3], L.u[0])

                else:
                    rhs.vel += L.dt * self.QI[m + 1, j] * self.get_full_f(f)

            # implicit solve with prefactor stemming from the diagonal of Qd

            L.u[m + 1] = rhs
            # update function values
            if self.coll.implicit:
                L.f[0] = P.eval_f(L.u[0], L.time)
                # L.f[1]=deepcopy(L.f[0])
                L.f[m + 1] = deepcopy(L.f[0])
            else:
                if m != 4:
                    L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        # indicate presence of new values at this level

        L.status.updated = True

        return None

    def compute_end_point(self):
        """
        In this Runge-Kutta implementation, the solution to the step is always stored in the last node
        """
        self.level.uend = self.level.u[-1]


class RKN(RungeKuttaNystrom):
    """
    Runge-Kutta-Nystrom method
    https://link.springer.com/book/10.1007/978-3-540-78862-1
    page: 284
    Chapter: II.14 Numerical methods for Second order differential equations
    """

    def __init__(self, params):
        nodes = np.array([0.0, 0.5, 0.5, 1])
        weights = np.array([1.0, 2.0, 2.0, 1.0]) / 6.0
        matrix = np.zeros([4, 4])
        matrix[1, 0] = 0.5
        matrix[2, 1] = 0.5
        matrix[3, 2] = 1.0

        weights_bar = np.array([1.0, 1.0, 1.0, 0]) / 6.0
        matrix_bar = np.zeros([4, 4])
        matrix_bar[1, 0] = 1 / 8
        matrix_bar[2, 0] = 1 / 8
        matrix_bar[3, 2] = 1 / 2
        params['butcher_tableau'] = ButcherTableau(weights, nodes, matrix)
        params['butcher_tableau_bar'] = ButcherTableau(weights_bar, nodes, matrix_bar)

        super(RKN, self).__init__(params)


class Velocity_Verlet(RungeKuttaNystrom):
    """
    Velocity-Verlet scheme
    https://de.wikipedia.org/wiki/Verlet-Algorithmus
    """

    def __init__(self, params):
        nodes = np.array([1.0, 1.0])
        weights = np.array([1 / 2, 0])
        matrix = np.zeros([2, 2])
        matrix[1, 1] = 1
        weights_bar = np.array([1 / 2, 0])
        matrix_bar = np.zeros([2, 2])
        params['butcher_tableau'] = ButcherTableau(weights, nodes, matrix)
        params['butcher_tableau_bar'] = ButcherTableau(weights_bar, nodes, matrix_bar)
        params['Velocity_verlet'] = True

        super(Velocity_Verlet, self).__init__(params)
