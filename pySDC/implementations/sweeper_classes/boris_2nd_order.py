import numpy as np

from pySDC.core.Sweeper import sweeper

import pdb
class boris_2nd_order(sweeper):
    """
    Custom sweeper class, implements Sweeper.py

    Second-order sweeper using velocity-Verlet with Boris scheme as base integrator

    Attributes:
        S: node-to-node collocation matrix (first order)
        SQ: node-to-node collocation matrix (second order)
        ST: node-to-node trapezoidal matrix
        Sx: node-to-node Euler half-step for position update
    """

    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """

        # call parent's initialization routine

        if "QI" not in params:
            params["QI"] = "IE"
        if "QE" not in params:
            params["QE"] = "EE"

        super(boris_2nd_order, self).__init__(params)

        # S- and SQ-matrices (derived from Q) and Sx- and ST-matrices for the integrator
        [
            self.S,
            self.ST,
            self.SQ,
            self.Sx,
            self.QQ,
            self.QI,
            self.QT,
            self.Qx,
            self.Q,
        ] = self.__get_Qd()

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

        # set implicit and explicit Euler matrices (default, but can be changed)
        QI = self.get_Qdelta_implicit(self.coll, qd_type=self.params.QI)
        QE = self.get_Qdelta_explicit(self.coll, qd_type=self.params.QE)

        # trapezoidal rule
        QT = 1 / 2 * (QI + QE)

        # Qx as in the paper
        Qx = np.dot(QE, QT) + 1 / 2 * QE * QE

        Sx = np.zeros(np.shape(self.coll.Qmat))
        ST = np.zeros(np.shape(self.coll.Qmat))
        S = np.zeros(np.shape(self.coll.Qmat))

        # fill-in node-to-node matrices
        Sx[0, :] = Qx[0, :]
        ST[0, :] = QT[0, :]
        S[0, :] = self.coll.Qmat[0, :]
        for m in range(self.coll.num_nodes):
            Sx[m + 1, :] = Qx[m + 1, :] - Qx[m, :]
            ST[m + 1, :] = QT[m + 1, :] - QT[m, :]
            S[m + 1, :] = self.coll.Qmat[m + 1, :] - self.coll.Qmat[m, :]
        # SQ via dot-product, could also be done via QQ
        SQ = np.dot(S, self.coll.Qmat)

        # QQ-matrix via product of Q
        QQ = np.dot(self.coll.Qmat, self.coll.Qmat)

        return [S, ST, SQ, Sx, QQ, QI, QT, Qx, self.coll.Qmat]

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a sieccf4e803b97ngle sweep over all nodes

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

        # initialize integral terms with zeros, will add stuff later
        integral = [P.dtype_u(P.init, val=0.0) for l in range(M)]

        # gather all terms which are known already (e.g. from the previous iteration)
        # this corresponds to SF(u^k) - SdF(u^k) + tau (note: have integrals in pos and vel!)
        for m in range(M):
            for j in range(M + 1):
                # build RHS from f-terms (containing the E field) and the B field
                f = P.build_f(L.f[j], L.u[j], L.time + L.dt * self.coll.nodes[j - 1])
                # add SQF(u^k) - SxF(u^k) for the position
                integral[m].pos += L.dt * (L.dt * (self.SQ[m + 1, j] - self.Sx[m + 1, j]) * f)
                # add SF(u^k) - STF(u^k) for the velocity
                integral[m].vel += L.dt * (self.S[m + 1, j] - self.ST[m + 1, j]) * f
            # add tau if associated
            if L.tau[m] is not None:
                integral[m] += L.tau[m]
                # tau is 0-to-node, need to change it to node-to-node here
                if m > 0:
                    integral[m] -= L.tau[m - 1]

        # do the sweep
        for m in range(0, M):
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            tmp = P.dtype_u(integral[m])
            for j in range(m + 1):
                # build RHS from f-terms (containing the E field) and the B field
                f = P.build_f(L.f[j], L.u[j], L.time + L.dt * self.coll.nodes[j - 1])
                # add SxF(u^{k+1})
                tmp.pos += L.dt * (L.dt * self.Sx[m + 1, j] * f)
            # add pos at previous node + dt*v0
            tmp.pos += L.u[m].pos + L.dt * self.coll.delta_m[m] * L.u[0].vel
            # set new position, is explicit
            L.u[m + 1].pos = tmp.pos

            # get E field with new positions and compute mean
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

            ck = tmp.vel

            # do the boris scheme
            L.u[m + 1].vel = P.boris_solver(ck, L.dt*np.diag(self.QI)[m+1] , L.f[m], L.f[m + 1], L.u[m])

        # indicate presence of new values at this level
        L.status.updated = True

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

            # integrate RHS over all collocation nodes, RHS is here only f(x,v)!
            for j in range(1, self.coll.num_nodes + 1):
                f = P.build_f(L.f[j], L.u[j], L.time + L.dt * self.coll.nodes[j - 1])
                p[-1].pos += L.dt * (L.dt * self.QQ[m, j] * f) + L.dt * self.coll.Qmat[m, j] * L.u[0].vel
                p[-1].vel += L.dt * self.coll.Qmat[m, j] * f

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
        L.uend = P.dtype_u(L.u[0])
        for m in range(self.coll.num_nodes):
            f = P.build_f(L.f[m + 1], L.u[m + 1], L.time + L.dt * self.coll.nodes[m])
            L.uend.pos += L.dt * (L.dt * self.qQ[m] * f) + L.dt * self.coll.weights[m] * L.u[0].vel
            L.uend.vel += L.dt * self.coll.weights[m] * f
        # add up tau correction of the full interval (last entry)
        if L.tau[-1] is not None:
            L.uend += L.tau[-1]

        return None

    def get_sweeper_mats(self):
        """
        Returns the matrices Q, QQ, Qx, QT which define the sweeper.
        """

        Q = self.Q[1:, 1:]
        QQ = self.QQ[1:, 1:]
        Qx = self.Qx[1:, 1:]
        QT = self.QT[1:, 1:]

        return Q, QQ, Qx, QT

    def get_scalar_problems_sweeper_mats(self, lambdas=None):
        """
        This function returns the corresponding matrices of an SDC sweep matrix formulation

        Args:
            lambdas (numpy.narray): the first entry in lambdas is k-spring constant and the second is mu friction.
        """

        Q, QQ, Qx, QT = self.get_sweeper_mats()

        if lambdas is None:
            pass
            # should use lambdas from attached problem and make sure it is scalar SDC
            raise NotImplementedError("At the moment, the values for the lambda have to be provided")
        else:
            k = lambdas[0]
            mu = lambdas[1]

        nnodes = self.coll.num_nodes
        dt = self.level.dt

        F = np.block(
            [
                [-k * np.eye(nnodes), -mu * np.eye(nnodes)],
                [-k * np.eye(nnodes), -mu * np.eye(nnodes)],
            ]
        )

        C_coll = np.block([[np.eye(nnodes), dt * Q], [np.zeros([nnodes, nnodes]), np.eye(nnodes)]])
        Q_coll = np.block(
            [
                [dt**2 * QQ, np.zeros([nnodes, nnodes])],
                [np.zeros([nnodes, nnodes]), dt * Q],
            ]
        )
        Q_vv = np.block(
            [
                [dt**2 * Qx, np.zeros([nnodes, nnodes])],
                [np.zeros([nnodes, nnodes]), dt * QT],
            ]
        )
        M_vv = np.eye(2 * nnodes) - np.dot(Q_vv, F)

        return C_coll, Q_coll, Q_vv, M_vv, F

    def get_scalar_problems_manysweep_mats(self, nsweeps, lambdas=None):

        """
        For a scalar problem, K sweeps of SDC can be written in matrix form.

        Args:
            nsweeps (int): number of sweeps
            lambdas (numpy.ndarray): the first entry in lambdas is k-spring constant and the second is mu friction.
        """
        nnodes = self.coll.num_nodes

        C_coll, Q_coll, Q_vv, M_vv, F = self.get_scalar_problems_sweeper_mats(lambdas=lambdas)

        K_sdc = np.dot(np.linalg.inv(M_vv), Q_coll - Q_vv) @ F

        Keig, Kvec = np.linalg.eig(K_sdc)

        Kp_sdc = np.linalg.matrix_power(K_sdc, nsweeps)
        Kinv_sdc = np.linalg.inv(np.eye(2 * nnodes) - K_sdc)

        Kdot_sdc = np.dot(np.eye(2 * nnodes) - Kp_sdc, Kinv_sdc)
        MC = np.dot(np.linalg.inv(M_vv), C_coll)

        Mat_sweep = Kp_sdc + np.dot(Kdot_sdc, MC)

        return Mat_sweep, np.max(np.abs(Keig))

    def get_scalar_problems_picardsweep_mats(self, nsweeps, lambdas=None):

        """
        For a scalar problem, K sweeps of SDC can be written in matrix form.

        Args:
            nsweeps (int): number of sweeps
            lambdas (numpy.ndarray): the first entry in lambdas is k-spring constant and the second is mu friction.
        """
        nnodes = self.coll.num_nodes

        C_coll, Q_coll, Q_vv, M_vv, F = self.get_scalar_problems_sweeper_mats(lambdas=lambdas)

        K_sdc = np.dot(Q_coll, F)

        Keig, Kvec = np.linalg.eig(K_sdc)

        Kp_sdc = np.linalg.matrix_power(K_sdc, nsweeps)
        Kinv_sdc = np.linalg.inv(np.eye(2 * nnodes) - K_sdc)

        Kdot_sdc = np.dot(np.eye(2 * nnodes) - Kp_sdc, Kinv_sdc)

        Mat_sweep = Kp_sdc + np.dot(Kdot_sdc, C_coll)

        return Mat_sweep, np.max(np.abs(Keig))
