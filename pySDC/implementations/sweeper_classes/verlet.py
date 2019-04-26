import numpy as np

from pySDC.core.Sweeper import sweeper
from pySDC.implementations.collocation_classes.gauss_lobatto import CollGaussLobatto


class verlet(sweeper):
    """
    Custom sweeper class, implements Sweeper.py

    Second-order sweeper using velocity-Verlet as base integrator

    Attributes:
        QQ: 0-to-node collocation matrix (second order)
        QT: 0-to-node trapezoidal matrix
        Qx: 0-to-node Euler half-step for position update
        qQ: update rule for final value (if needed)
    """

    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """

        if 'QI' not in params:
            params['QI'] = 'IE'
        if 'QE' not in params:
            params['QE'] = 'EE'

        # call parent's initialization routine
        super(verlet, self).__init__(params)

        # Trapezoidal rule, Qx and Double-Q as in the Boris-paper
        [self.QT, self.Qx, self.QQ] = self.__get_Qd()

        self.qQ = np.dot(self.coll.weights, self.coll.Qmat[1:, 1:])

    def __get_Qd(self):
        """
        Get integration matrices for 2nd-order SDC

        Returns:
            S: node-to-node collocation matrix (first order)
            SQ: node-to-node collocation matrix (second order)
            ST: node-to-node trapezoidal matrix
            Sx: node-to-node Euler half-step for position update
        """

        # set implicit and explicit Euler matrices
        QI = self.get_Qdelta_implicit(self.coll, self.params.QI)
        QE = self.get_Qdelta_explicit(self.coll, self.params.QE)

        # trapezoidal rule
        QT = 0.5 * (QI + QE)
        # QT = QI

        # Qx as in the paper
        Qx = np.dot(QE, QT) + 0.5 * QE * QE

        QQ = np.zeros(np.shape(self.coll.Qmat))

        # if we have Gauss-Lobatto nodes, we can do a magic trick from the Book
        # this takes Gauss-Lobatto IIIB and create IIIA out of this
        if isinstance(self.coll, CollGaussLobatto):

            for m in range(self.coll.num_nodes):
                for n in range(self.coll.num_nodes):
                    QQ[m + 1, n + 1] = self.coll.weights[n] * (1.0 - self.coll.Qmat[n + 1, m + 1] /
                                                               self.coll.weights[m])
            QQ = np.dot(self.coll.Qmat, QQ)

        # if we do not have Gauss-Lobatto, just multiply Q (will not get a symplectic method, they say)
        else:

            QQ = np.dot(self.coll.Qmat, self.coll.Qmat)

        return [QT, Qx, QQ]

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
        # get QF(u^k)
        integral = self.integrate()
        for m in range(M):

            # get -QdF(u^k)_m
            for j in range(1, M + 1):
                integral[m].pos -= L.dt * (L.dt * self.Qx[m + 1, j] * L.f[j])
                integral[m].vel -= L.dt * self.QT[m + 1, j] * L.f[j]

            # add initial value
            integral[m].pos += L.u[0].pos
            integral[m].vel += L.u[0].vel
            # add tau if associated
            if L.tau[m] is not None:
                integral[m] += L.tau[m]

        # do the sweep
        for m in range(0, M):
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            L.u[m + 1] = P.dtype_u(integral[m])
            for j in range(1, m + 1):
                # add QxF(u^{k+1})
                L.u[m + 1].pos += L.dt * (L.dt * self.Qx[m + 1, j] * L.f[j])
                L.u[m + 1].vel += L.dt * self.QT[m + 1, j] * L.f[j]

            # get RHS with new positions
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

            L.u[m + 1].vel += L.dt * self.QT[m + 1, m + 1] * L.f[m + 1]

        # indicate presence of new values at this level
        L.status.updated = True

        # # do the sweep (alternative description)
        # for m in range(0, M):
        #     # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
        #     L.u[m + 1] = P.dtype_u(integral[m])
        #     for j in range(1, m + 1):
        #         # add QxF(u^{k+1})
        #         L.u[m + 1].pos += L.dt * (L.dt * self.Qx[m + 1, j] * L.f[j])
        #
        #     # get RHS with new positions
        #     L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])
        #
        # for m in range(0, M):
        #     for n in range(0, M):
        #         L.u[m + 1].vel += L.dt * self.QT[m + 1, n + 1] * L.f[n + 1]
        #
        # # indicate presence of new values at this level
        # L.status.updated = True

        return None

    def integrate(self):
        """
        Integrates the right-hand side

        Returns:
            list of dtype_u: containing the integral as values
        """

        # get current level and problem description
        L = self.level
        P = L.prob
        # create new instance of dtype_u, initialize values with 0
        p = []
        for m in range(1, self.coll.num_nodes + 1):
            p.append(P.dtype_u(P.init, val=0.0))

            # integrate RHS over all collocation nodes, RHS is here only f(x)!
            for j in range(1, self.coll.num_nodes + 1):
                p[-1].pos += L.dt * (L.dt * self.QQ[m, j] * L.f[j]) + L.dt * self.coll.Qmat[m, j] * L.u[0].vel
                p[-1].vel += L.dt * self.coll.Qmat[m, j] * L.f[j]
                # we need to set mass and charge here, too, since the code uses the integral to create new particles
                p[-1].m = L.u[0].m
                p[-1].q = L.u[0].q

        return p

    def compute_end_point(self):
        """
        Compute u at the right point of the interval

        The value uend computed here is a full evaluation of the Picard formulation (always!)

        Returns:
            None
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # start with u0 and add integral over the full interval (using coll.weights)
        if (self.coll.right_is_node and not self.params.do_coll_update):
            # a copy is sufficient
            L.uend = P.dtype_u(L.u[-1])
        else:
            L.uend = P.dtype_u(L.u[0])
            for m in range(self.coll.num_nodes):
                L.uend.pos += L.dt * (L.dt * self.qQ[m] * L.f[m + 1]) + L.dt * self.coll.weights[m] * L.u[0].vel
                L.uend.vel += L.dt * self.coll.weights[m] * L.f[m + 1]
                # remember to set mass and charge here, too
                L.uend.m = L.u[0].m
                L.uend.q = L.u[0].q
            # add up tau correction of the full interval (last entry)
            if L.tau[-1] is not None:
                L.uend += L.tau[-1]

        return None
