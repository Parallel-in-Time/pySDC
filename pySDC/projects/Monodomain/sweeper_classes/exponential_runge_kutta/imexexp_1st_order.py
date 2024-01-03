import numpy as np

from pySDC.core.Sweeper import sweeper
from pySDC.core.Errors import CollocationError, ParameterError
import numdifftools.fornberg as fornberg


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

        self.lmbda = None

        self.lambda_and_phi_outdated = True
        # compute weights w such that PiQ^(k)(0) = sum_{j=0}^{M-1} w[k,j]*Q[j], k=0,...,M-1
        M = self.coll.num_nodes
        c = self.coll.nodes
        self.w = fornberg.fd_weights_all(c, 0.0, M - 1)

    def compute_lambda_and_phi(self):
        if self.lambda_and_phi_outdated:
            L = self.level
            P = L.prob
            M = self.coll.num_nodes
            c = self.coll.nodes

            # compute lambda
            self.lmbda = P.lmbda_eval(L.u[0], L.time)

            # improve stability? yes but doesnt converge
            # for i in range(self.lmbda.size):
            #     self.lmbda.val_list[i].values[:] = self.lmbda.val_list[i].values.min()
            # ---

            # another try
            # for i in range(M):
            #     lmbda_i=P.lmbda_eval(L.u[i + 1], L.time + c[i] * L.dt)
            #     for j in range(self.lmbda.size):
            #         self.lmbda.val_list[j].values[:]=np.minimum(self.lmbda.val_list[j].values, lmbda_i.val_list[j].values)
            # ----
            # another
            # self.lmbda *= 1.5
            # ----

            if not hasattr(self, "phi"):
                self.phi = [[P.dtype_u(init=P.init, val=0.0) for i in range(M)] for k in range(M + 1)]
                self.phi_one = [[P.dtype_u(init=P.init, val=0.0) for i in range(M)]]
            self.phi = P.phi_eval_lists(L.u[0], L.dt * c, L.time, list(range(M + 1)), phi=self.phi, lmbda=self.lmbda, update_non_exp_indeces=False)
            self.phi_one = P.phi_eval_lists(L.u[0], L.dt * self.delta, L.time, [1], phi=self.phi_one, lmbda=self.lmbda, update_non_exp_indeces=True)

            # compute weight for the integration of \int_0^ci exp(dt*(ci-r)lmbda)*PiQ(r)dr = \sum_{j=0}^{M-1} Qmat_exp[i,j]*Q[j]
            if not hasattr(self, "Qmat_exp"):
                self.Qmat_exp = [[P.dtype_u(init=P.init, val=0.0) for j in range(M)] for i in range(M)]
            for i in range(M):
                for j in range(M):
                    self.Qmat_exp[i][j].zero_sub(P.rhs_exp_indeces)
                    for k in range(M):
                        self.Qmat_exp[i][j].axpy_sub(self.w[k, j] * c[i] ** (k + 1), self.phi[k + 1][i], P.rhs_exp_indeces)
                        # self.Qmat_exp[i][j] += self.w[k, j] * c[i] ** (k + 1) * self.phi[k + 1][i]

            self.lambda_and_phi_outdated = False

    def integrate(self):
        """
        Integrates the right-hand side (here impl + expl + exp)

        Returns:
            list of dtype_u: containing the integral as values
        """

        # get current level and problem description
        L = self.level
        P = L.prob
        M = self.coll.num_nodes

        self.compute_lambda_and_phi()

        if not hasattr(self, "Q"):
            self.Q = [P.dtype_u(init=P.init, val=0.0) for _ in range(M)]
            self.tmp = P.dtype_u(init=P.init, val=0.0)

        # compute polynomial. remember that L.u[k+1] corresponds to c[k]
        for k in range(M):
            # self.Q[k] = L.f[k + 1].exp + self.lmbda * (L.u[0] - L.u[k + 1])  # at the indeces of the exponential rhs, otherwsie 0
            self.Q[k].zero_sub(P.rhs_exp_indeces)
            self.Q[k].iadd_sub(L.u[0], P.rhs_exp_indeces)
            self.Q[k].axpy_sub(-1.0, L.u[k + 1], P.rhs_exp_indeces)
            self.Q[k].imul_sub(self.lmbda, P.rhs_exp_indeces)
            self.Q[k].iadd_sub(L.f[k + 1].exp, P.rhs_exp_indeces)

        # integrate RHS over all collocation nodes
        me = [P.dtype_u(init=P.init, val=0.0) for _ in range(M)]
        for m in range(1, self.coll.num_nodes + 1):
            for j in range(1, self.coll.num_nodes + 1):
                # me[m - 1] += self.coll.Qmat[m, j] * (L.f[j].impl + L.f[j].expl) + self.Qmat_exp[m - 1][j - 1] * (self.Q[j - 1])
                me[m - 1].axpy_sub(self.coll.Qmat[m, j], L.f[j].impl, P.rhs_stiff_indeces)
                me[m - 1].axpy_sub(self.coll.Qmat[m, j], L.f[j].expl, P.rhs_nonstiff_indeces)
                self.tmp.copy_sub(self.Q[j - 1], P.rhs_exp_indeces)
                self.tmp.imul_sub(self.Qmat_exp[m - 1][j - 1], P.rhs_exp_indeces)
                me[m - 1].iadd_sub(self.tmp, P.rhs_exp_indeces)
            me[m - 1] *= L.dt

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

        # do the sweep, method 1
        for m in range(M):
            # integral[m] -= L.dt * self.delta[m] * (L.f[m].expl + L.f[m + 1].impl + self.phi_one[0][m] * (L.f[m].exp + self.lmbda * (L.u[0] - L.u[m])))
            integral[m].axpy_sub(-L.dt * self.delta[m], L.f[m + 1].impl, P.rhs_stiff_indeces)
            integral[m].axpy_sub(-L.dt * self.delta[m], L.f[m].expl, P.rhs_nonstiff_indeces)
            self.tmp.copy_sub(L.u[0], P.rhs_exp_indeces)
            self.tmp.axpy_sub(-1.0, L.u[m], P.rhs_exp_indeces)
            self.tmp.imul_sub(self.lmbda, P.rhs_exp_indeces)
            self.tmp.iadd_sub(L.f[m].exp, P.rhs_exp_indeces)
            self.tmp.imul_sub(self.phi_one[0][m], P.rhs_exp_indeces)
            integral[m].axpy_sub(-L.dt * self.delta[m], self.tmp, P.rhs_exp_indeces)
        for m in range(M):
            # rhs = L.u[m] + integral[m] + L.dt * self.delta[m] * (L.f[m].expl + self.phi_one[0][m] * (L.f[m].exp + self.lmbda * (L.u[0] - L.u[m])))
            self.tmp.zero()
            self.tmp.copy_sub(L.u[0], P.rhs_exp_indeces)
            self.tmp.axpy_sub(-1.0, L.u[m], P.rhs_exp_indeces)
            self.tmp.imul_sub(self.lmbda, P.rhs_exp_indeces)
            self.tmp.iadd_sub(L.f[m].exp, P.rhs_exp_indeces)
            self.tmp.imul_sub(self.phi_one[0][m], P.rhs_exp_indeces)
            self.tmp.iadd_sub(L.f[m].expl, P.rhs_nonstiff_indeces)
            self.tmp.aypx(L.dt * self.delta[m], integral[m])
            self.tmp += L.u[m]

            # implicit solve with prefactor stemming from QI
            L.u[m + 1] = P.solve_system(self.tmp, L.dt * self.QI[m + 1, m + 1], L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

            if L.u[m + 1].is_nan_or_inf():
                L.u[m + 1] = L.u[m].copy()

            # update function values
            P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m], fh=L.f[m + 1])

            # for robustness we check that we are not exploding
            if (
                (L.f[m + 1].impl.is_nan_or_inf() or abs(L.f[m + 1].impl) > 1e6 * (abs(L.f[m].impl) + 1))
                or (L.f[m + 1].expl.is_nan_or_inf() or abs(L.f[m + 1].expl) > 1e6 * (abs(L.f[m].expl) + 1))
                or (L.f[m + 1].exp.is_nan_or_inf() or abs(L.f[m + 1].exp) > 1e6 * (abs(L.f[m].exp) + 1))
            ):
                L.u[m + 1] = L.u[m].copy()
                L.f[m + 1].expl = L.f[m].expl.copy()
                L.f[m + 1].impl = L.f[m].impl.copy()
                L.f[m + 1].exp = L.f[m].exp.copy()

        # do the sweep, method 2
        # for m in range(M):
        #     rhs = L.u[m] + L.dt * self.delta[m] * phi_one[m] * (L.f[m].exp + lmbda * (L.u[0] - L.u[m]))
        #     P.eval_f(rhs, L.time + L.dt * self.coll.nodes[m], eval_impl=False, eval_expl=True, eval_exp=False, fh=L.f[m + 1])
        #     integral[m] -= L.dt * self.delta[m] * (L.f[m + 1].expl + L.f[m + 1].impl) + (rhs - L.u[m])
        # for m in range(M):
        #     rhs = L.u[m] + L.dt * self.delta[m] * phi_one[m] * (L.f[m].exp + lmbda * (L.u[0] - L.u[m]))

        #     P.eval_f(rhs, L.time + L.dt * self.coll.nodes[m], eval_impl=False, eval_expl=True, eval_exp=False, fh=L.f[m + 1])
        #     rhs += L.dt * self.delta[m] * L.f[m + 1].expl + integral[m]

        #     # implicit solve with prefactor stemming from QI
        #     L.u[m + 1] = P.solve_system(rhs, L.dt * self.QI[m + 1, m + 1], L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        #     # update function values
        #     L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

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

        return None

    def predict(self):
        """
        Predictor to fill values at nodes before first sweep

        Default prediction for the sweepers, only copies the values to all collocation nodes
        and evaluates the RHS of the ODE there
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # evaluate RHS at left point
        L.f[0] = P.eval_f(L.u[0], L.time)

        for m in range(1, self.coll.num_nodes + 1):
            # copy u[0] to all collocation nodes, evaluate RHS
            if self.params.initial_guess == "spread":
                L.u[m] = P.dtype_u(L.u[0])
                L.f[m] = P.eval_f(L.u[m], L.time + L.dt * self.coll.nodes[m - 1])
            # start with zero everywhere
            elif self.params.initial_guess == "zero":
                L.u[m] = P.dtype_u(init=P.init, val=0.0)
                L.f[m] = P.dtype_f(init=P.init, val=0.0)
            # start with random initial guess
            elif self.params.initial_guess == "random":
                L.u[m] = P.dtype_u(init=P.init, val=self.rng.rand(1)[0])
                L.f[m] = P.dtype_f(init=P.init, val=self.rng.rand(1)[0])
            else:
                raise ParameterError(f"initial_guess option {self.params.initial_guess} not implemented")

        # indicate that this level is now ready for sweeps
        L.status.unlocked = True
        L.status.updated = True

        self.update_lmbda_yinf_status(outdated=True)

    def update_lmbda_yinf_status(self, outdated):
        if not self.level.prob.constant_lambda_and_phi and outdated:
            self.lambda_and_phi_outdated = True

    def compute_residual(self):
        """
        Computation of the residual using the collocation matrix Q
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # check if there are new values (e.g. from a sweep)
        # assert L.status.updated

        # compute the residual for each node

        # build QF(u)
        res_norm = []
        rel_res_norm = []
        res = self.integrate()
        for m in range(self.coll.num_nodes):
            res[m] += L.u[0] - L.u[m + 1]
            # add tau if associated
            if L.tau[m] is not None:
                res[m] += L.tau[m]
            # use abs function from data type here
            res_norm.append(abs(res[m]))
            rel_res_norm.append(res[m].rel_norm(L.u[0]))

        # find maximal residual over the nodes
        if L.params.residual_type == 'full_abs':
            L.status.residual = max(res_norm)
        elif L.params.residual_type == 'last_abs':
            L.status.residual = res_norm[-1]
        elif L.params.residual_type == 'full_rel':
            L.status.residual = max(rel_res_norm)
        elif L.params.residual_type == 'last_rel':
            L.status.residual = rel_res_norm[-1]
        else:
            raise ParameterError(f'residual_type = {L.params.residual_type} not implemented, choose ' f'full_abs, last_abs, full_rel or last_rel instead')

        # indicate that the residual has seen the new values
        L.status.updated = False

        return None
