import numpy as np

from pySDC.core.Sweeper import sweeper
from pySDC.core.Errors import CollocationError, ParameterError
import numdifftools.fornberg as fornberg

from pySDC.projects.Monodomain.sweeper_classes.exponential_runge_kutta.imexexp_1st_order import imexexp_1st_order


class imexexp_1st_order_mass(imexexp_1st_order):
    """
    Custom sweeper class, implements Sweeper.py

    First-order IMEXEXP sweeper using implicit/explicit/exponential Euler as base integrator
    In the cardiac electrphysiology community this is known as Rush-Larsen scheme.

    Attributes:
        QI: implicit Euler integration matrix
        QE: explicit Euler integration matrix
    """

    def __init__(self, params):
        super(imexexp_1st_order_mass, self).__init__(params)

    def integrate(self):
        L = self.level
        P = L.prob
        M = self.coll.num_nodes

        me = super().integrate()
        me_mass = [P.apply_mass_matrix(me[m]) for m in range(M)]

        return me_mass

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
            integral[m] -= P.apply_mass_matrix(L.dt * self.delta[m] * (L.f[m].expl + L.f[m + 1].impl + self.phi_one[0][m] * (L.f[m].exp + self.lmbda * (L.u[0] - L.u[m]))))
            # integral[m].axpy_sub(-L.dt * self.delta[m], L.f[m + 1].impl, P.rhs_stiff_indeces)
            # integral[m].axpy_sub(-L.dt * self.delta[m], L.f[m].expl, P.rhs_nonstiff_indeces)
            # self.tmp.copy_sub(L.u[0], P.rhs_exp_indeces)
            # self.tmp.axpy_sub(-1.0, L.u[m], P.rhs_exp_indeces)
            # self.tmp.imul_sub(self.lmbda, P.rhs_exp_indeces)
            # self.tmp.iadd_sub(L.f[m].exp, P.rhs_exp_indeces)
            # self.tmp.imul_sub(self.phi_one[0][m], P.rhs_exp_indeces)
            # integral[m].axpy_sub(-L.dt * self.delta[m], self.tmp, P.rhs_exp_indeces)
        for m in range(M):
            rhs = integral[m] + P.apply_mass_matrix(L.u[m] + L.dt * self.delta[m] * (L.f[m].expl + self.phi_one[0][m] * (L.f[m].exp + self.lmbda * (L.u[0] - L.u[m]))))
            # self.tmp.zero()
            # self.tmp.copy_sub(L.u[0], P.rhs_exp_indeces)
            # self.tmp.axpy_sub(-1.0, L.u[m], P.rhs_exp_indeces)
            # self.tmp.imul_sub(self.lmbda, P.rhs_exp_indeces)
            # self.tmp.iadd_sub(L.f[m].exp, P.rhs_exp_indeces)
            # self.tmp.imul_sub(self.phi_one[0][m], P.rhs_exp_indeces)
            # self.tmp.iadd_sub(L.f[m].expl, P.rhs_nonstiff_indeces)
            # self.tmp.aypx(L.dt * self.delta[m], integral[m])
            # self.tmp += L.u[m]

            # implicit solve with prefactor stemming from QI
            L.u[m + 1] = P.solve_system(rhs, L.dt * self.QI[m + 1, m + 1], L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

            # update function values
            P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m], fh=L.f[m + 1])

        # print('dopo update')
        # for m in range(M + 1):
        #     print(f"norms(L.u[{m}]) = {abs(L.u[m])}")

        L.status.updated = True

        return None

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
        Mu0 = P.apply_mass_matrix(L.u[0])
        for m in range(self.coll.num_nodes):
            res[m] += P.apply_mass_matrix(L.u[0] - L.u[m + 1])
            # add tau if associated
            if L.tau[m] is not None:
                res[m] += L.tau[m]
            # use abs function from data type here
            res_norm.append(abs(res[m]))
            rel_res_norm.append(res[m].rel_norm(Mu0))

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
