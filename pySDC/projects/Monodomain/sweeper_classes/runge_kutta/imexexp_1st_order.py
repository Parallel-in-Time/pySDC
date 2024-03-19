import numpy as np

from pySDC.core.Sweeper import sweeper
from pySDC.core.Errors import CollocationError


class imexexp_1st_order(sweeper):
    """
    Custom sweeper class, implements Sweeper.py

    First-order IMEXEXP sweeper using implicit/explicit/exponential Euler as base integrator
    In the cardiac electrphysiology community this is known as Rush-Larsen scheme.

    Attributes:
        QI: implicit Euler integration matrix
        QE: explicit Euler integration matrix
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
        self.QI = self.get_Qdelta_implicit(coll=self.coll, qd_type=self.params.QI)
        self.delta = np.diagonal(self.QI)[1:]

    def eval_phi_f_exp(self, u, factor):
        L = self.level
        P = L.prob
        self.lmbda = P.dtype_u(init=P.init, val=0.0)
        self.yinf = P.dtype_u(init=P.init, val=0.0)
        P.eval_lmbda_yinf_exp(u, self.lmbda, self.yinf)
        phi_f_exp = P.dtype_u(init=P.init, val=0.0)
        for i in P.rhs_exp_indeces:
            phi_f_exp.np_array(i)[:] = u.np_array(i)[:] - self.yinf.np_array(i)[:]
            phi_f_exp.np_array(i)[:] *= (np.exp(factor * self.lmbda.np_array(i)) - 1.0) / factor

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
            # new instance of dtype_u, initialize values with 0
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

        # do the sweep: expl and exp at same u
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
            raise CollocationError("This option is not implemented yet.")
            # start with u0 and add integral over the full interval (using coll.weights)
            L.uend = P.dtype_u(L.u[0])
            for m in range(self.coll.num_nodes):
                L.uend += L.dt * self.coll.weights[m] * (L.f[m + 1].impl + L.f[m + 1].expl)
            # add up tau correction of the full interval (last entry)
            if L.tau[-1] is not None:
                L.uend += L.tau[-1]

        return None
