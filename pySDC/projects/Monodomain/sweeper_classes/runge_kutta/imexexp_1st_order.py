import numpy as np

from pySDC.core.sweeper import Sweeper
from pySDC.core.errors import CollocationError


class imexexp_1st_order(Sweeper):
    """
    Custom sweeper class, implements Sweeper.py

    First-order IMEXEXP sweeper using implicit/explicit/exponential Euler as base integrator
    In the cardiac electrphysiology community this is known as Rush-Larsen scheme.

    """

    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """

        if "QI" not in params:
            params["QI"] = "IE"

        # call parent's initialization routine
        super(imexexp_1st_order, self).__init__(params)

        # IMEX integration matrices
        self.QI = self.get_Qdelta_implicit(qd_type=self.params.QI)
        self.delta = np.diagonal(self.QI)[1:]

    def eval_phi_f_exp(self, u, factor):
        """
        Evaluates the exponential part of the right-hand side f_exp(u)=lambda(u)*(u-y_inf(u)) multiplied by the exponential factor phi_1(factor*lambda)
        Since phi_1(z)=(e^z-1)/z then phi_1(factor*lambda) * f_exp(u) = ((e^(factor*lambda)-1)/factor) *(u-y_inf(u))
        """
        L = self.level
        P = L.prob
        self.lmbda = P.dtype_u(init=P.init, val=0.0)
        self.yinf = P.dtype_u(init=P.init, val=0.0)
        P.eval_lmbda_yinf_exp(u, self.lmbda, self.yinf)
        phi_f_exp = P.dtype_u(init=P.init, val=0.0)
        for i in P.rhs_exp_indeces:
            phi_f_exp[i][:] = u[i] - self.yinf[i][:]
            phi_f_exp[i][:] *= (np.exp(factor * self.lmbda[i]) - 1.0) / factor

        return phi_f_exp

    def integrate(self):
        """
        Integrates the right-hand side (here impl + expl + exp)

        Returns:
            list of dtype_u: containing the integral as values
        """

        # get current level and problem description
        L = self.level

        me = []

        # integrate RHS over all collocation nodes
        for m in range(1, self.coll.num_nodes + 1):
            me.append(L.dt * self.coll.Qmat[m, 1] * (L.f[1].impl + L.f[1].expl + L.f[1].exp))
            for j in range(2, self.coll.num_nodes + 1):
                me[m - 1] += L.dt * self.coll.Qmat[m, j] * (L.f[j].impl + L.f[j].expl + L.f[j].exp)

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

        integral = self.integrate()
        for m in range(M):
            if L.tau[m] is not None:
                integral[m] += L.tau[m]
        for i in range(1, M):
            integral[M - i] -= integral[M - i - 1]

        # do the sweep
        for m in range(M):
            integral[m] -= (
                L.dt
                * self.delta[m]
                * (L.f[m].expl + L.f[m + 1].impl + self.eval_phi_f_exp(L.u[m], L.dt * self.delta[m]))
            )
        for m in range(M):
            rhs = (
                L.u[m]
                + integral[m]
                + L.dt * self.delta[m] * (L.f[m].expl + self.eval_phi_f_exp(L.u[m], L.dt * self.delta[m]))
            )

            # implicit solve with prefactor stemming from QI
            L.u[m + 1] = P.solve_system(rhs, L.dt * self.delta[m], L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

            # update function values
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        # indicate presence of new values at this level
        L.status.updated = True

        return None

    def compute_end_point(self):
        """
        Compute u at the right point of the interval

        The value uend computed here is a full evaluation of the Picard formulation unless do_full_update==False

        Returns:
            None
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # check if Mth node is equal to right point and do_coll_update is false, perform a simple copy
        if self.coll.right_is_node and not self.params.do_coll_update:
            # a copy is sufficient
            L.uend = P.dtype_u(L.u[-1])
        else:
            raise CollocationError(
                "In this sweeper we expect the right point to be a collocation node and do_coll_update==False"
            )

        return None
