import numpy as np

from pySDC.core.Sweeper import sweeper
from pySDC.core.Errors import CollocationError
from pySDC.projects.ExplicitStabilized.explicit_stabilized_classes.rho_estimator import rho_estimator
from pySDC.core.Lagrange import LagrangeApproximation
from pySDC.projects.ExplicitStabilized.explicit_stabilized_classes.es_methods import mRKC1
import numdifftools.fornberg as fornberg
from dolfinx import io


class exponential_multirate_explicit_stabilized(sweeper):
    """
    Custom sweeper class, implements Sweeper.py

    First-order stabilized exponential sweeper using explicit stabilized methods and exponential Euler as base integrators

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
        super(exponential_multirate_explicit_stabilized, self).__init__(params)

        self.QI = self.get_Qdelta_implicit(coll=self.coll, qd_type=self.params.QI)
        # self.delta = self.coll.delta_m
        self.delta = np.diagonal(self.QI)[1:]

        self.damping = params["damping"] if "damping" in params else 0.05
        self.safe_add = params["safe_add"] if "safe_add" in params else 1

        self.s_forced = params["es_s"] if "es_s" in params else 0
        self.m_forced = params["es_m"] if "es_m" in params else 0

        self.scale_separation = True
        self.es = self.init_es(params)
        self.s_prev = [0] * self.coll.num_nodes
        self.m_prev = [0] * self.coll.num_nodes

        self.sub_nodes = [None] * self.coll.num_nodes
        self.interp_mat = [None] * self.coll.num_nodes
        self.interp_mat0 = [None] * self.coll.num_nodes

        self.nodes = self.coll.nodes
        self.nodes0 = np.array([0.0])
        self.nodes0 = np.append(self.nodes0, self.nodes)
        self.lagrange = LagrangeApproximation(self.nodes)
        self.lagrange0 = LagrangeApproximation(self.nodes0)

        # updates spectral radius every rho_freq steps
        self.rho_freq = params["rho_freq"] if "rho_freq" in params else 5
        self.rho_count = 0

        # self.with_node_zero = True

    def init_es(self, params):
        return [mRKC1(params["es_class_outer"], params["es_class_inner"], self.damping, self.safe_add, self.scale_separation) for _ in range(self.coll.num_nodes)]

    def update_stages_coefficients(self):
        L = self.level
        P = L.prob
        M = self.coll.num_nodes

        if not hasattr(self, "rho_expl"):
            self.estimated_rho = [0, 0]
            if hasattr(P.exact, "rho_nonstiff") and callable(P.exact.rho_nonstiff):
                self.rho_expl = P.exact.rho_nonstiff
            else:
                self.rho_estimator = rho_estimator(P)
                self.rho_expl = self.rho_estimator.rho_expl
            if hasattr(P.exact, "rho_stiff") and callable(P.exact.rho_stiff):
                self.rho_impl = P.exact.rho_stiff
            else:
                if not hasattr(self, "rho_estimator"):
                    self.rho_estimator = rho_estimator(P)
                self.rho_impl = self.rho_estimator.rho_impl

        if self.rho_count % self.params.rho_freq == 0:
            self.estimated_rho[0] = self.rho_expl(y=L.u[0], t=L.time, fy=L.f[0])
            self.estimated_rho[1] = self.rho_impl(y=L.u[0], t=L.time, fy=L.f[0])
            self.rho_count = 0
            self.s = [0] * M
            self.m = [0] * M
            for m in range(M):
                rho = self.estimated_rho
                self.s[m] = self.es[m].get_s(L.dt * self.delta[m] * rho[0]) if self.s_forced == 0 else self.s_forced
                self.m[m] = self.es[m].get_m(L.dt * self.delta[m], rho[1], self.s[m]) if self.m_forced == 0 else self.m_forced
                if self.m_forced > 0:
                    self.es[m].fix_eta(L.dt, self.s[m], self.m[m])
                if self.s[m] != self.s_prev[m] or self.m[m] != self.m_prev[m]:
                    self.es[m].update_coefficients(self.s[m], self.m[m])
                    self.sub_nodes[m] = []
                    self.interp_mat[m] = []
                    self.interp_mat0[m] = []
                    for j in range(self.s[m]):
                        self.sub_nodes[m].append(
                            self.coll.nodes[m] - self.delta[m] + (self.es[m].c[j] - self.es[m].delta_c[j]) * self.delta[m] + (self.es[m].eta / L.dt) * (self.es[m].d - self.es[m].delta_d)
                        )
                        self.interp_mat[m].append(self.lagrange.getInterpolationMatrix(self.sub_nodes[m][j]))
                        self.interp_mat0[m].append(self.lagrange0.getInterpolationMatrix(self.sub_nodes[m][j]))
            self.s_prev = self.s[:]
            self.m_prev = self.m[:]

        self.rho_count += 1

    def integrate(self):
        """
        Integrates the right-hand side (here impl + expl + exp)

        Returns:
            list of dtype_u: containing the integral as values
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        me = []

        M = self.coll.num_nodes

        # this works for M==1
        # lmbda = P.lmbda_eval(L.u[0],L.time)
        # phi_one = P.phi_eval(L.u[0],L.dt,L.time,1)
        # me.append( L.dt*( L.f[1].impl + L.f[1].expl +  phi_one.__rmul__(L.f[1].exp+lmbda.__rmul__(L.u[0]-L.u[1]))) )
        # return me

        # compute phi[k][i] = phi_{k}(dt*c_i*lmbda), i=0,...,M-1, k=0,...,M
        phi = []
        c = self.coll.nodes
        for k in range(M + 1):
            phi.append([])
            for i in range(M):
                phi[k].append(P.phi_eval(L.u[0], L.dt * c[i], L.time, k))

        # compute polynomial. remember that L.u[k+1] corresponds to c[k]
        lmbda = P.lmbda_eval(L.u[0], L.time)
        Q = []
        for k in range(M):
            Q.append(L.f[k + 1].exp + lmbda * (L.u[0] - L.u[k + 1]))

        # compute weights w such that PiQ^(k)(0) = sum_{j=0}^{M-1} w[k,j]*Q[j], k=0,...,M-1
        w = fornberg.fd_weights_all(c, 0.0, M - 1)

        # compute weight for the integration of \int_0^ci exp(dt*(ci-r)lmbda)*PiQ(r)dr = \sum_{j=0}^{M-1} Qmat_exp[i,j]*Q[j]
        Qmat_exp = [[P.dtype_u(P.init, val=0.0) for j in range(M)] for i in range(M)]
        for i in range(M):
            for j in range(M):
                for k in range(M):
                    Qmat_exp[i][j] += (w[k, j] * c[i] ** (k + 1)) * phi[k + 1][i]

        # integrate RHS over all collocation nodes
        for m in range(1, self.coll.num_nodes + 1):
            me.append(L.dt * (self.coll.Qmat[m, 1] * (L.f[1].impl + L.f[1].expl) + Qmat_exp[m - 1][0] * (Q[0])))
            # new instance of dtype_u, initialize values with 0
            for j in range(2, self.coll.num_nodes + 1):
                me[m - 1] += L.dt * (self.coll.Qmat[m, j] * (L.f[j].impl + L.f[j].expl) + Qmat_exp[m - 1][j - 1] * (Q[j - 1]))

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

        self.update_stages_coefficients()

        integral = self.integrate()
        for m in range(M):
            integral[m] += L.u[0]
            if L.tau[m] is not None:
                integral[m] += L.tau[m]

        u_old = [P.dtype_u(u) for u in L.u]

        integral.insert(0, L.u[0])

        self.logger.debug(f"s = {[es.s for es in self.es]}, m = {[es.m for es in self.es]}, rho = {self.estimated_rho}")

        self.dj = P.dtype_u(P.init, 0.0)
        self.djm1 = P.dtype_u(P.init, 0.0)
        self.djm2 = P.dtype_u(P.init, 0.0)

        def print_list_norms(lst, name):
            for i, l in enumerate(lst):
                print(f"|{name}[{i}]|={abs(l)}")

        # do the sweep
        for m in range(M):
            dt = L.dt * self.delta[m]
            es = self.es[m]

            self.djm1.copy(self.dj)
            df = self.eval_df_eta(self.djm1, integral, u_old, L.time + L.dt * (self.coll.nodes[m] - self.delta[m]), m, 0)
            self.dj = self.djm1 + dt * es.mu[0] * (df)
            for j in range(2, es.s + 1):
                self.djm2, self.djm1, self.dj = self.djm1, self.dj, self.djm2

                df = self.eval_df_eta(self.djm1, integral, u_old, L.time + L.dt * (self.coll.nodes[m] - self.delta[m] + es.c[j - 2] * self.delta[m]), m, j - 1)
                self.dj = df
                self.dj *= dt * es.mu[j - 1]
                self.dj += es.nu[j - 1] * self.djm1 + es.kappa[j - 1] * self.djm2

            L.u[m + 1] = self.dj + integral[m + 1]
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        # # indicate presence of new values at this level
        L.status.updated = True

        return None

    def interpolate(self, val, m, j, k):
        interp_mat = self.interp_mat0[m][j]
        I_val = interp_mat[k, 0] * val[0]
        for i in range(1, len(val)):
            I_val += interp_mat[k, i] * val[i]

        return I_val

    def eval_df_eta(self, d, integral, u, t, m, j):
        L = self.level
        P = L.prob
        es = self.es[m]

        I_int = self.interpolate(integral, m, j, 0)
        I_u = self.interpolate(u, m, j, 0)

        f1 = P.eval_f(I_int + d, t, eval_impl=False, eval_expl=False, eval_exp=True)
        f2 = P.eval_f(I_u, t, eval_impl=False, eval_expl=False, eval_exp=True)
        lmbda = P.lmbda_eval(L.u[0], L.time)
        df_exp = P.phi_eval(L.u[0], es.eta * es.alpha[0], t, 1) * (f1.exp - f2.exp + lmbda * (I_u - (I_int + d)))
        djm1 = d + es.eta * df_exp

        f1 = P.eval_f(I_int + djm1, t, eval_impl=True, eval_expl=True, eval_exp=False)
        f2 = P.eval_f(I_u, t, eval_impl=True, eval_expl=True, eval_exp=False)
        df_expl = f1.expl - f2.expl
        df_impl = f1.impl - f2.impl

        dj = djm1 + es.eta * es.alpha[0] * (df_impl + df_expl)
        djm2 = P.dtype_u(P.init, 0.0)
        for k in range(2, es.m + 1):
            djm2, djm1, dj = djm1, dj, djm2

            I_int = self.interpolate(integral, m, j, k - 1)
            I_u = self.interpolate(u, m, j, k - 1)
            f1 = P.eval_f(I_int + djm1, t + es.eta * es.d[k - 2], eval_impl=True, eval_expl=False, eval_exp=False)
            f2 = P.eval_f(I_u, t + es.eta * es.d[k - 2], eval_impl=True, eval_expl=False, eval_exp=False)
            df_impl = f1.impl - f2.impl

            dj = es.beta[k - 1] * djm1 + es.gamma[k - 1] * djm2 + es.eta * es.alpha[k - 1] * (df_expl + df_impl)

        dj -= d
        dj *= 1.0 / es.eta

        return dj

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
