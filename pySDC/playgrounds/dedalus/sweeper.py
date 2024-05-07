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

    def predict(self):

        L = self.level
        t0, dt, wall_time = L.time, L.dt, 0.0

        P:DedalusProblem = L.prob
        assert type(P) == DedalusProblem
        P.firstEval = True

        Fk, LXk = P.F[0], P.LX[0]

        P.evalLX(LXk[0])
        P.evalF(Fk[0], t0, dt, wall_time)
        for m in range(1, self.coll.num_nodes):
            np.copyto(LXk[m].data, LXk[0].data)
            np.copyto(Fk[m].data, Fk[0].data)

        # indicate that this level is now ready for sweeps
        L.status.unlocked = True
        L.status.updated = True


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
            L.u[m+1] = P.stateCopy()

            # Evaluate and store LX with current state
            P.evalLX(LXk1[m])
            # Evaluate and store F(X, t) with current state
            P.evalF(Fk1[m], t0+dt*tau[m], dt, wall_time)

        # Inverse position for iterate k and k+1 in storage
        # ie making the new evaluation the old for next iteration
        P.F.rotate()
        P.LX.rotate()

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
