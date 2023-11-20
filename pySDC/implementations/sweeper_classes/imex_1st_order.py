import numpy as np

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit


class imex_1st_order(generic_implicit):
    """
    First-order IMEX sweeper using implicit/explicit Euler as base integrator

    Attributes:
        QI: implicit Euler integration matrix
        QE: explicit Euler integration matrix
    """

    def __init__(self, params):
        if 'QE' not in params:
            params['QE'] = 'EE'

        super().__init__(params)

        self.QE = self.get_Qdelta_explicit(coll=self.coll, qd_type=self.params.QE)

    def add_Q_minus_QD_times_F(self, rhs):
        self.add_matrix_times_implicit_f_evaluations_to(self.coll.Qmat - self.QI, rhs)
        self.add_matrix_times_explicit_f_evaluations_to(self.coll.Qmat - self.QE, rhs)

    def add_matrix_times_implicit_f_evaluations_to(self, matrix, rhs):
        for m in range(1, self.coll.num_nodes + 1):
            for j in range(1, self.coll.num_nodes + 1):
                rhs[m - 1] += self.level.dt * matrix[m, j] * self.level.f[j].impl

    def add_matrix_times_explicit_f_evaluations_to(self, matrix, rhs):
        for m in range(1, self.coll.num_nodes + 1):
            for j in range(1, self.coll.num_nodes + 1):
                rhs[m - 1] += self.level.dt * matrix[m, j] * self.level.f[j].expl

    def add_new_information_from_forward_substitution(self, rhs, current_node):
        for j in range(1, current_node + 1):
            rhs[current_node] += self.level.dt * self.QI[current_node + 1, j] * self.level.f[j].impl
            rhs[current_node] += self.level.dt * self.QE[current_node + 1, j] * self.level.f[j].expl

    def integrate(self):
        """
        Integrates the right-hand side

        Returns:
            list of dtype_u: containing the integral as values
        """
        integral = self.initialize_right_hand_side_buffer()
        self.add_matrix_times_implicit_f_evaluations_to(self.coll.Qmat, integral)
        self.add_matrix_times_explicit_f_evaluations_to(self.coll.Qmat, integral)
        return integral

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
