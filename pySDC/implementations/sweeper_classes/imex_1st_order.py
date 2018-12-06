import numpy as np
import scipy.linalg
from pySDC.core.Sweeper import sweeper


class imex_1st_order(sweeper):
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

        if 'QI' not in params:
            params['QI'] = 'IE'
        if 'QE' not in params:
            params['QE'] = 'EE'

        # call parent's initialization routine
        super(imex_1st_order, self).__init__(params)

        # IMEX integration matrices
        self.QI = self.get_Qdelta_implicit(coll=self.coll, qd_type=self.params.QI)
        self.QE = self.get_Qdelta_explicit(coll=self.coll, qd_type=self.params.QE)

    def integrate(self):
        """
        Integrates the right-hand side (here impl + expl)

        Returns:
            list of dtype_u: containing the integral as values
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        me = []

        # integrate RHS over all collocation nodes
        for m in range(1, self.coll.num_nodes + 1):
            # new instance of dtype_u, initialize values with 0
            me.append(P.dtype_u(P.init, val=0))
            for j in range(1, self.coll.num_nodes + 1):
                me[-1] += L.dt * self.coll.Qmat[m, j] * (L.f[j].impl + L.f[j].expl)

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
        for m in range(M):
            # subtract QIFI(u^k)_m + QEFE(u^k)_m
            for j in range(M + 1):
                integral[m] -= L.dt * (self.QI[m + 1, j] * L.f[j].impl + self.QE[m + 1, j] * L.f[j].expl)
            # add initial value
            integral[m] += L.u[0]
            # add tau if associated
            if L.tau[m] is not None:
                integral[m] += L.tau[m]

        # do the sweep
        for m in range(0, M):
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            rhs = P.dtype_u(integral[m])
            for j in range(m + 1):
                rhs += L.dt * (self.QI[m + 1, j] * L.f[j].impl + self.QE[m + 1, j] * L.f[j].expl)

            # implicit solve with prefactor stemming from QI
            L.u[m + 1] = P.solve_system(rhs, L.dt * self.QI[m + 1, m + 1], L.u[m + 1],
                                        L.time + L.dt * self.coll.nodes[m])
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
            # start with u0 and add integral over the full interval (using coll.weights)
            L.uend = P.dtype_u(L.u[0])
            for m in range(self.coll.num_nodes):
                L.uend += L.dt * self.coll.weights[m] * (L.f[m + 1].impl + L.f[m + 1].expl)
            # add up tau correction of the full interval (last entry)
            if L.tau[-1] is not None:
                L.uend += L.tau[-1]

        return None

    def get_sweeper_mats(self):
        """
        Returns the three matrices Q, QI, QE which define the sweeper.

        The first row and column, corresponding to the left starting value, are removed to correspond to the notation
        introduced in Ruprecht & Speck, Spectral deferred corrections with fast-wave slow-wave splitting, 2016
        """
        QE = self.QE[1:, 1:]
        QI = self.QI[1:, 1:]
        Q = self.coll.Qmat[1:, 1:]
        return QE, QI, Q

    def get_scalar_problems_sweeper_mats(self, lambdas=None):
        """
        This function returns the corresponding matrices of an IMEX-SDC sweep in matrix formulation.

        See Ruprecht & Speck, Spectral deferred corrections with fast-wave slow-wave splitting, 2016 for the derivation.

        Args:
            lambdas (numpy.ndarray): the first entry in lambdas is lambda_fast, the second is lambda_slow.
        """
        QE, QI, Q = self.get_sweeper_mats()
        if lambdas is None:
            pass
            # should use lambdas from attached problem and make sure it is a scalar IMEX
            raise NotImplementedError("At the moment, the values for lambda have to be provided")
        else:
            lambda_fast = lambdas[0]
            lambda_slow = lambdas[1]
        nnodes = self.coll.num_nodes
        dt = self.level.dt
        LHS = np.eye(nnodes) - dt * (lambda_fast * QI + lambda_slow * QE)
        RHS = dt * ((lambda_fast + lambda_slow) * Q - (lambda_fast * QI + lambda_slow * QE))
        return LHS, RHS

    def get_scalar_problems_manysweep_mat(self, nsweeps, lambdas=None):
        """
        For a scalar problem, K sweeps of IMEX-SDC can be written in matrix form.

        Args:
            nsweeps (int): number of sweeps
            lambdas (numpy.ndarray): the first entry in lambdas is lambda_fast, the second is lambda_slow.
        """
        LHS, RHS = self.get_scalar_problems_sweeper_mats(lambdas=lambdas)
        Pinv = np.linalg.inv(LHS)
        Mat_sweep = np.linalg.matrix_power(Pinv.dot(RHS), nsweeps)
        for k in range(0, nsweeps):
            Mat_sweep += np.linalg.matrix_power(Pinv.dot(RHS), k).dot(Pinv)
        return Mat_sweep
