#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweeper class for dedalus
"""
import numpy as np

from problem import DedalusProblem
from pySDC.core.Sweeper import sweeper

class DedalusSweeperIMEX(sweeper):

    def __init__(self, params):
        if 'QI' not in params: params['QI'] = 'IE'
        if 'QE' not in params: params['QE'] = 'EE'
        # call parent's initialization routine
        super().__init__(params)
        # IMEX integration matrices
        self.QI = self.get_Qdelta_implicit(coll=self.coll, qd_type=self.params.QI)
        self.QE = self.get_Qdelta_explicit(coll=self.coll, qd_type=self.params.QE)


    def update_nodes(self):
        """
        TODO
        """
        # get current level and problem description
        L = self.level
        P:DedalusProblem = L.prob
        assert type(P) == DedalusProblem
        P.firstEval = True
        solver = P.solver

        # only if the level has been touched before
        assert L.status.unlocked

        # get number of collocation nodes for easier access
        M = self.coll.num_nodes
        assert M == P.M

        solver.state = L.u[0]


        # Attribute references
        tau, qI, qE, q = self.coll.nodes, self.QI[1:, 1:], self.QE[1:, 1:], self.coll.Qmat[1:, 1:]
        t0, dt, wall_time = L.time, L.dt, 0.0
        RHS, MX0 = P.RHS, P.MX0
        Fk, LXk, Fk1, LXk1 = P.F[0], P.LX[0], P.F[1], P.LX[1]
        axpy = P.axpy

        P.updateLHS(dt, qI)
        P.computeMX0(solver.state, MX0)

        # Loop on all quadrature nodes
        for m in range(M):

            # Initialize with MX0 term
            np.copyto(RHS.data, MX0.data)

            # Add quadrature terms
            for j in range(M):
                axpy(a=dt*q[m, j], x=Fk[j].data, y=RHS.data)
                axpy(a=-dt*q[m, j], x=LXk[j].data, y=RHS.data)

            # Add F and LX terms from iteration k+1
            for j in range(m):
                axpy(a=dt*qE[m, j], x=Fk1[j].data, y=RHS.data)
                axpy(a=-dt*qI[m, j], x=LXk1[j].data, y=RHS.data)

            # Add F and LX terms from iteration k
            for j in range(m):
                axpy(a=-dt*qE[m, j], x=Fk[j].data, y=RHS.data)
                axpy(a=dt*qI[m, j], x=LXk[j].data, y=RHS.data)
            axpy(a=dt*qI[m, m], x=LXk[m].data, y=RHS.data)

            # Solve system and store node solution in solver state
            P.solveAndStoreState(m)
            L.u[m] = P.stateCopy()

            # Evaluate and store LX with current state
            P.evalLX(LXk1[m])
            # Evaluate and store F(X, t) with current state
            P.evalF(Fk1[m], t0+dt*tau[m], dt, wall_time)

        # Inverse position for iterate k and k+1 in storage
        # ie making the new evaluation the old for next iteration
        P.F.rotate()
        P.LX.rotate()

        # # gather all terms which are known already (e.g. from the previous iteration)
        # # this corresponds to u0 + QF(u^k) - QIFI(u^k) - QEFE(u^k) + tau

        # # get QF(u^k)
        # integral = self.integrate()
        # for m in range(M):
        #     # subtract QIFI(u^k)_m + QEFE(u^k)_m
        #     for j in range(1, M + 1):
        #         integral[m] -= L.dt * (self.QI[m + 1, j] * L.f[j].impl + self.QE[m + 1, j] * L.f[j].expl)
        #     # add initial value
        #     integral[m] += L.u[0]
        #     # add tau if associated
        #     if L.tau[m] is not None:
        #         integral[m] += L.tau[m]

        # # do the sweep
        # for m in range(0, M):
        #     # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
        #     rhs = P.dtype_u(integral[m])
        #     for j in range(1, m + 1):
        #         rhs += L.dt * (self.QI[m + 1, j] * L.f[j].impl + self.QE[m + 1, j] * L.f[j].expl)

        #     # implicit solve with prefactor stemming from QI
        #     L.u[m + 1] = P.solve_system(
        #         rhs, L.dt * self.QI[m + 1, m + 1], L.u[m + 1], L.time + L.dt * self.coll.nodes[m]
        #     )

        #     # update function values
        #     L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        # indicate presence of new values at this level
        L.status.updated = True


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
            L.uend = [f.copy() for f in L.u[-1]]
        else:
            raise NotImplementedError()
            # start with u0 and add integral over the full interval (using coll.weights)
            L.uend = P.dtype_u(L.u[0])
            for m in range(self.coll.num_nodes):
                L.uend += L.dt * self.coll.weights[m] * (L.f[m + 1].impl + L.f[m + 1].expl)
            # add up tau correction of the full interval (last entry)
            if L.tau[-1] is not None:
                L.uend += L.tau[-1]

        return None
