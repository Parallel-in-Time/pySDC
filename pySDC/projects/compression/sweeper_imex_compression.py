import numpy as np
from pySDC.projects.compression.CRAM_Manager import CRAM_Manager
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order


class imex_1st_order_compression(imex_1st_order):
    """
    Custom sweeper class, implements Sweeper.py

    First-order IMEX sweeper using implicit/explicit Euler as base integrator

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

        # call parent's initialization routine
        super().__init__(params)

        # Instantiate CRAM_Manager class
        self.manager = CRAM_Manager("ABS", "sz", 100)
        self.manager_register = False

    def predict(self):
        super().predict()
        # Register all the variables
        if self.manager_register == False:
            self.manager_register = True
            self.mgr_list = ["u", "fi", "fe"]
            for i in self.mgr_list:
                self.manager.registerVar(
                    i,
                    self.level.prob.init,
                    self.level.prob.init[2],
                    numVectors=self.coll.num_nodes + 1,
                    errBoundMode="ABS",
                    compType="sz3",
                    errBound=1e-5,
                )
        # TODO: remove dtype, edit it

        # IMEX integration matrices

    def integrate(self):
        """
        Integrates the right-hand side (here impl + expl)

        Returns:
            list of dtype_u: containing the integral as values
        """

        # get current level and problem description
        L = self.level

        me = []

        # integrate RHS over all collocation nodes
        for m in range(1, self.coll.num_nodes + 1):
            me.append(L.dt * self.coll.Qmat[m, 1] * (L.f[1].impl + L.f[1].expl))
            # new instance of dtype_u, initialize values with 0
            for j in range(2, self.coll.num_nodes + 1):
                me[m - 1] += L.dt * self.coll.Qmat[m, j] * (L.f[j].impl + L.f[j].expl)

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
        integral = self.integrate()
        for jj in range(M + 1):
            self.manager.compress(L.f[jj].impl, "fi", jj)
            self.manager.compress(L.f[jj].expl, "fe", jj)
            self.manager.compress(L.u[jj], "u", jj)

        u_zero = self.manager.decompress("u", 0)

        for m in range(M):
            # subtract QIFI(u^k)_m + QEFE(u^k)_m
            for j in range(1, M + 1):
                f_impl = self.manager.decompress("fi", j)
                f_expl = self.manager.decompress("fe", j)
                integral[m] -= L.dt * (self.QI[m + 1, j] * f_impl + self.QE[m + 1, j] * f_expl)
            # add initial value
            integral[m] += u_zero
            # add tau if associated
            if L.tau[m] is not None:
                integral[m] += L.tau[m]

        # do the sweep
        for m in range(0, M):
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            rhs = P.dtype_u(integral[m])
            for j in range(1, m + 1):
                f_impl = self.manager.decompress("fi", j)
                f_expl = self.manager.decompress("fe", j)
                rhs += L.dt * (self.QI[m + 1, j] * f_impl + self.QE[m + 1, j] * f_expl)

            # implicit solve with prefactor stemming from QI
            u_prev = self.manager.decompress("u", m + 1)
            u_sol = P.solve_system(
                rhs,
                L.dt * self.QI[m + 1, m + 1],
                u_prev,
                L.time + L.dt * self.coll.nodes[m],
            )
            # update function values
            # L.f[m+1].impl = self.manager.decompress('fi',m+1)
            # L.f[m+1].expl = self.manager.decompress('fe',m+1)
            f_sol = P.eval_f(u_sol, L.time + L.dt * self.coll.nodes[m])

            self.manager.compress(u_sol, "u", m + 1)
            self.manager.compress(f_sol.impl, "fi", m + 1)
            self.manager.compress(f_sol.expl, "fe", m + 1)
            # L.u[m+1][:] = self.manager.decompress('u',m+1)
            # print(m+1,abs(L.u[m+1]-u_sol), abs(L.u[m+1]))
            # self.manager.compress(L.f[m + 1].impl,'fi',m+1)
            # self.manager.compress(L.f[m + 1].expl,'fe',m+1)

        # indicate presence of new values at this level
        L.status.updated = True
        for m in range(M + 1):
            #     self.manager.compress(L.u[m + 1],'u',m+1)
            #     self.manager.compress(L.f[m + 1].impl,'fi',m+1)
            #     self.manager.compress(L.f[m + 1].expl,'fe',m+1)
            L.u[m] = self.manager.decompress("u", m)
            L.f[m].impl[:] = self.manager.decompress("fi", m)
            L.f[m].expl[:] = self.manager.decompress("fe", m)

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
            # start with u0 and add integral over the full interval (using coll.weights)
            L.uend = P.dtype_u(L.u[0])
            for m in range(self.coll.num_nodes):
                L.uend += L.dt * self.coll.weights[m] * (L.f[m + 1].impl + L.f[m + 1].expl)
            # add up tau correction of the full interval (last entry)
            if L.tau[-1] is not None:
                L.uend += L.tau[-1]

        return None
