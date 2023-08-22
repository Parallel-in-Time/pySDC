import numpy as np
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.core.Errors import ParameterError


class efficient_sweeper:
    """
    Replace the predict function by something that does not excessively evaluate superfluous right hand sides.
    """

    def predict(self):
        """
        Predictor to fill values at nodes before first sweep. Skip evaluation of RHS on initial conditions.

        Default prediction for the sweepers, only copies the values to all collocation nodes
        and evaluates the RHS of the ODE there
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        for m in range(1, self.coll.num_nodes + 1):
            # copy u[0] to all collocation nodes, evaluate RHS
            if self.params.initial_guess == 'spread':
                L.u[m] = P.dtype_u(L.u[0])
                L.f[m] = P.eval_f(L.u[m], L.time + L.dt * self.coll.nodes[m - 1])
            # start with zero everywhere
            elif self.params.initial_guess == 'zero':
                L.u[m] = P.dtype_u(init=P.init, val=0.0)
                L.f[m] = P.dtype_f(init=P.init, val=0.0)
            # start with random initial guess
            elif self.params.initial_guess == 'random':
                L.u[m] = P.dtype_u(init=P.init, val=self.rng.rand(1)[0])
                L.f[m] = P.dtype_f(init=P.init, val=self.rng.rand(1)[0])
            else:
                raise ParameterError(f'initial_guess option {self.params.initial_guess} not implemented')

        # indicate that this level is now ready for sweeps
        L.status.unlocked = True
        L.status.updated = True


class generic_implicit_efficient(efficient_sweeper, generic_implicit):
    """
    This sweeper has the same functionality of the `generic_implicit` sweeper, but saves a few operations at the expense
    of readability.
    """

    def integrate(self, Q=None):
        """
        Integrates the right-hand side. Depending on `Q`, this may or may not be consistent with an integral
        approximation.

        Args:
            Q (numpy.ndarray): Some sort of quadrature rule

        Returns:
            list of dtype_u: containing the integral as values
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        Q = self.coll.Qmat if Q is None else Q

        me = []

        # integrate RHS over all collocation nodes
        for m in range(1, self.coll.num_nodes + 1):
            # new instance of dtype_u, initialize values with 0
            me.append(P.dtype_u(P.init, val=0.0))
            for j in range(1, self.coll.num_nodes + 1):
                me[-1] += L.dt * Q[m, j] * L.f[j]

        return me

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes

        Returns:
            None
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked

        # get number of collocation nodes for easier access
        M = self.coll.num_nodes

        # gather all terms which are known already (e.g. from the previous iteration)
        # this corresponds to u0 + QF(u^k) - QdF(u^k) + tau

        # get QF(u^k)
        integral = self.integrate(Q=self.coll.Qmat - self.QI)
        for m in range(M):
            # add initial value
            integral[m] += L.u[0]
            # add tau if associated
            if L.tau[m] is not None:
                integral[m] += L.tau[m]

        # do the sweep
        for m in range(0, M):
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            rhs = P.dtype_u(integral[m])
            for j in range(1, m + 1):
                rhs += L.dt * self.QI[m + 1, j] * L.f[j]

            # implicit solve with prefactor stemming from the diagonal of Qd
            L.u[m + 1] = P.solve_system(
                rhs, L.dt * self.QI[m + 1, m + 1], L.u[m + 1], L.time + L.dt * self.coll.nodes[m]
            )
            # update function values
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        # indicate presence of new values at this level
        L.status.updated = True

        return None

    def compute_residual(self, stage=''):
        if stage not in self.params.skip_residual_computation:
            lvl = self.level
            lvl.f[-1] = lvl.prob.eval_f(lvl.u[-1], lvl.time + lvl.dt * self.coll.nodes[-1])
        super().compute_residual(stage)


class imex_1st_order_efficient(efficient_sweeper, imex_1st_order):
    """
    Duplicate of `imex_1st_order` sweeper which is slightly more efficient at the cost of code readability.
    """

    def integrate(self, Q=None, QI=None, QE=None):
        """
        Integrates the right-hand side (here impl + expl)

        Args:
            Q (numpy.ndarray): Full quadrature rule
            QI (numpy.ndarray): Implicit preconditioner
            QE (numpy.ndarray): Explicit preconditioner

        Returns:
            list of dtype_u: containing the integral as values
        """

        Q = self.coll.Qmat if Q is None else Q
        QI = np.zeros_like(Q) if QI is None else QI
        QE = np.zeros_like(Q) if QE is None else QE

        # get current level and problem description
        L = self.level

        me = []

        # integrate RHS over all collocation nodes
        for m in range(1, self.coll.num_nodes + 1):
            me.append(L.dt * ((Q - QI)[m, 1] * L.f[1].impl + (Q - QE)[m, 1] * L.f[1].expl))
            # new instance of dtype_u, initialize values with 0
            for j in range(2, self.coll.num_nodes + 1):
                me[m - 1] += L.dt * ((Q - QI)[m, j] * L.f[j].impl + (Q - QE)[m, j] * L.f[j].expl)

        return me

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes

        Returns:
            None
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked

        # get number of collocation nodes for easier access
        M = self.coll.num_nodes

        # gather all terms which are known already (e.g. from the previous iteration)
        # this corresponds to u0 + QF(u^k) - QIFI(u^k) - QEFE(u^k) + tau

        # get QF(u^k)
        integral = self.integrate(Q=self.coll.Qmat, QI=self.QI, QE=self.QE)
        for m in range(M):
            # add initial value
            integral[m] += L.u[0]
            # add tau if associated
            if L.tau[m] is not None:
                integral[m] += L.tau[m]

        # do the sweep
        for m in range(0, M):
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            rhs = P.dtype_u(integral[m])
            for j in range(1, m + 1):
                rhs += L.dt * (self.QI[m + 1, j] * L.f[j].impl + self.QE[m + 1, j] * L.f[j].expl)

            # implicit solve with prefactor stemming from QI
            L.u[m + 1] = P.solve_system(
                rhs, L.dt * self.QI[m + 1, m + 1], L.u[m + 1], L.time + L.dt * self.coll.nodes[m]
            )

            # update function values
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        # indicate presence of new values at this level
        L.status.updated = True

        return None
