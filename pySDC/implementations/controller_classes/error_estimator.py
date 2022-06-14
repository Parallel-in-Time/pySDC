import numpy as np
from scipy.special import factorial

from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh
from pySDC.core.Errors import DataError


class _ErrorEstimatorBase:
    """

    This class should be the parent of all error estimator classes, MPI and nonMPI and provide all functions that can
    be shared.

    """

    def __init__(self, controller, order, size):
        self.params = controller.params

        if self.params.use_extrapolation_estimate:
            self.setup_extrapolation(controller, order, size)

    def setup_extrapolation(self, controller, order, size):
        """
        The extrapolation based method requires storage of previous values of u, f, t and dt and also requires solving
        a linear system of equations to compute the Taylor expansion finite difference style. Here, all variables are
        initialized which are needed for this process.
        """
        # check if we can handle the parameters
        if not controller.MS[0].levels[0].sweep.coll.right_is_node:
            raise NotImplementedError('I don\'t know what to do if the last collocation node is not the end point')

        # determine the order of the Taylor expansion to be higher than that of the time marching scheme
        if self.params.use_HotRod:
            self.order = order - 1 + 2
        else:
            self.order = order + 2

        # important: the variables to store the solutions etc. are defined in the children classes
        self.n = (self.order + 1) // 2  # since we store u and f, we need only half of each (the +1 is for rounding)
        self.n_per_proc = int(np.ceil(self.n / size))  # number of steps that each step needs to store
        self.u_coeff = [None] * self.n
        self.f_coeff = [0.] * self.n

    def communicate_time(self):
        raise NotImplementedError('Please implement a function to communicate the time and step sizes!')

    def communicate(self):
        raise NotImplementedError('Please implement a function to communicates the solution etc.!')

    def get_extrapolation_coefficients(self, t_eval=None):
        """
        This function solves a linear system where in the matrix A, the row index reflects the order of the derivative
        in the Taylor expansion and the column index reflects the particular step and whether its u or f from that
        step. The vector b on the other hand, contains a 1 in the first entry and zeros elsewhere, since we want to
        compute the value itself and all the derivatives should vanish after combining the Taylor expansions. This
        works to the order the number of rows and since we want a square matrix for solving, we need the same amount of
        colums, which determines the memory overhead, since it is equal to the solutions / rhs that we need in memory
        at the time of evaluation.
        This is enough to get the extrapolated solution, but if we want to compute the local error, we have to compute
        a prefactor. This is based on error accumulation between steps (first step's solution is exact plus 1 LTE,
        second solution is exact plus 2 LTE and so on), which can be computed for adaptive step sizes as well, but its
        wonky in time-parallel versions to say the least (it's not cared for and hence wrong, but it might be wrong in
        the same way as the embedded method and work for Hot Rod regardless...)
        """
        t, dt = self.communicate_time()

        # prepare A matrix
        A = np.zeros((self.order, self.order))
        A[0, 0:self.n] = 1.
        j = np.arange(self.order)
        inv_facs = 1. / factorial(j)

        # get the steps backwards from the point of evaluation
        idx = np.argsort(t)
        if t_eval is None:
            steps_from_now = -np.cumsum(dt[idx][::-1])[self.n - 1::-1]
        else:
            steps_from_now = t[idx] - t_eval

        # fill A matrix
        for i in range(1, self.order):
            # Taylor expansions of the solutions
            A[i, :self.n] = steps_from_now**j[i] * inv_facs[i]

            # Taylor expansions of the first derivatives a.k.a. right hand side evaluations
            A[i, self.n:self.order] = steps_from_now[2 * self.n - self.order:]**(j[i] - 1) * inv_facs[i - 1]

        # prepare rhs
        b = np.zeros(self.order)
        b[0] = 1.

        # solve linear system for the coefficients
        coeff = np.linalg.solve(A, b)
        self.u_coeff = coeff[:self.n]
        self.f_coeff[self.n * 2 - self.order:] = coeff[self.n:self.order]  # indexing takes care of uneven order

        # determine prefactor
        r = abs(self.dt[len(self.dt) - len(self.u_coeff):] / self.dt[-1])**(self.order - 1)
        inv_prefactor = -sum(r[1:]) - 1.
        for i in range(len(self.u_coeff)):
            inv_prefactor += sum(r[1: i + 1]) * self.u_coeff[i]
        self.prefactor = 1. / abs(inv_prefactor)

    def store_values(self, S):
        """
        Store the required attributes of the step to do the extrapolation. We only care about the last collocation
        node on the finest level at the moment.
        """
        if self.params.use_extrapolation_estimate:
            # figure out which values are to be replaced by the new ones
            if None in self.t:
                oldest_val = len(self.t) - len(self.t[self.t == [None]])
            else:
                oldest_val = np.argmin(self.t)

            f = S.levels[0].f[-1]
            if type(f) == imex_mesh:
                self.f[oldest_val] = f.impl + f.expl
            elif type(f) == mesh:
                self.f[oldest_val] = f
            else:
                raise DataError(f'Unable to store f from datatype {type(f)}, extrapolation based error estimate only\
                works with types imex_mesh and mesh')

            self.u[oldest_val] = S.levels[0].u[-1]
            self.t[oldest_val] = S.time + S.dt
            self.dt[oldest_val] = S.dt

    def embedded_estimate(self, S):
        """
        Compute embedded error estimate on the last node of each level
        In serial this is the local error, but in block Gauss-Seidel MSSDC this is a semi-global error in each block
        """
        for L in S.levels:
            # order rises by one between sweeps, making this so ridiculously easy
            L.status.error_embedded_estimate = max([abs(L.uold[-1] - L.u[-1]), np.finfo(float).eps])

    def extrapolation_estimate(self, S):
        """
        The extrapolation estimate combines values of u and f from multiple steps to extrapolate and compare to the
        solution obtained by the time marching scheme.
        """
        if None not in self.dt:
            if None in self.u_coeff or self.params.use_adaptivity:
                self.get_extrapolation_coefficients(t_eval=S.time + S.dt)

            self.communicate()

            if len(S.levels) > 1:
                raise NotImplementedError('Extrapolated estimate only works on the finest level for now')

            u_ex = S.levels[0].u[-1] * 0.
            idx = np.argsort(self.t)

            # see if we need to leave out any values because we are doing something in a block
            if (abs(S.time + S.dt - self.t) < 10. * np.finfo(float).eps).any():
                idx_step = idx[np.argmin(abs(self.t - S.time - S.dt))]
            else:
                idx_step = max(idx) + 1
            mask = np.logical_and(idx < idx_step, idx >= idx_step - self.n)

            for i in range(self.n):
                u_ex += self.u_coeff[i] * self.u[idx[mask][i]] + self.f_coeff[i] * self.f[idx[mask][i]]

            S.levels[0].status.error_extrapolation_estimate = abs(u_ex - S.levels[0].u[-1]) * self.prefactor

    def estimate(self, S):
        if self.params.use_HotRod:
            if S.status.iter == S.params.maxiter - 1:
                self.extrapolation_estimate(S)
            elif S.status.iter == S.params.maxiter:
                self.embedded_estimate(S)

        else:
            # only estimate errors when last sweep is performed and not when doing Hot Rod
            if S.status.iter == S.params.maxiter:

                if self.params.use_extrapolation_estimate:
                    self.extrapolation_estimate(S)

                if self.params.use_embedded_estimate or self.params.use_adaptivity:
                    self.embedded_estimate(S)


class _ErrorEstimator_nonMPI_BlockGS(_ErrorEstimatorBase):
    """

    Error estimator that works with the non-MPI controller in block Gauss-Seidel mode

    """

    def __init__(self, controller):
        super(_ErrorEstimator_nonMPI_BlockGS, self).__init__(controller, order=controller.MS[0].params.maxiter,
                                                             size=len(controller.MS))

    def store_values(self, MS):
        for S in MS:
            super(_ErrorEstimator_nonMPI_BlockGS, self).store_values(S)

    def communicate_time(self):
        return self.t, self.dt

    def communicate(self):
        pass

    def estimate(self, MS):
        # loop in reverse through the block since later steps lag behind with iterations
        for i in range(len(MS) - 1, -1, -1):
            S = MS[i]
            if self.params.use_HotRod:
                if S.status.iter == S.params.maxiter - 1:
                    self.extrapolation_estimate(S)
                elif S.status.iter == S.params.maxiter:
                    self.embedded_estimate_local_error(MS[:i + 1])
                    break

            else:
                # only estimate errors when last sweep is performed and not when doing Hot Rod
                if S.status.iter == S.params.maxiter:

                    if self.params.use_extrapolation_estimate:
                        self.extrapolation_estimate(S)

                    if self.params.use_embedded_estimate or self.params.use_adaptivity:
                        self.embedded_estimate_local_error(MS[:i + 1])

    def setup_extrapolation(self, controller, order, size):
        super(_ErrorEstimator_nonMPI_BlockGS, self).setup_extrapolation(controller, order, size)

        # check if we fixed the order by fixing the iteration number
        if not controller.MS[0].levels[0].params.restol < 0:
            raise NotImplementedError('Extrapolation based error estimate so far only with fixed order!')

        # check if we have the same order everywhere
        maxiter = [controller.MS[i].params.maxiter for i in range(len(controller.MS))]
        if not maxiter.count(maxiter[0]) == len(maxiter):
            raise NotImplementedError('All steps need to have the same order in time!')
        if controller.params.mssdc_jac:
            raise NotImplementedError('Extrapolation error only implemented in block Gauss-Seidel!')

        # check if we can deal with the supplied number of processes
        if len(controller.MS) > 1 and len(controller.MS) < self.n + 1:
            raise NotImplementedError(f'Extrapolation error estimate only works in serial, or in a no-overhead version\
 which requires at least {self.n+1} processes for order {self.order} Taylor expansion. You gave {size} processes.')

        # create variables to store u, f, t and dt from previous steps
        self.u = [None] * self.n_per_proc * size
        self.f = [None] * self.n_per_proc * size
        self.t = np.array([None] * self.n_per_proc * size)
        self.dt = np.array([None] * self.n_per_proc * size)

    def embedded_estimate_local_error(self, MS):
        """
        In block Gauss-Seidel SDC, the embedded estimate actually estimates sort of the global error within the block,
        since the second to last sweep is from an entirely k-1 order method, so to speak. This means the regular
        embedded method here yields this semi-global error and we get the local error as the difference of consecutive
        semi-global errors.
        """
        # prepare a list to store all errors in
        semi_global_errors = np.array([[0.] * len(MS[0].levels)] * (len(MS) + 1))

        for i in range(len(MS)):
            S = MS[i]
            for j in range(len(S.levels)):
                L = S.levels[j]
                semi_global_errors[i][j] = abs(L.uold[-1] - L.u[-1])
                L.status.error_embedded_estimate = max([abs(semi_global_errors[i][j] - semi_global_errors[i - 1][j]),
                                                       np.finfo(float).eps])


class _ErrorEstimator_nonMPI_no_memory_overhead_BlockGS(_ErrorEstimator_nonMPI_BlockGS):
    """

    Error estimator that works with the non-MPI controller in block Gauss-Seidel mode and does not feature memory
    overhead due to extrapolation error estimates, since the required values are in memory of other "processes"
    anyways.

    """

    def __init__(self, controller):
        super(_ErrorEstimator_nonMPI_no_memory_overhead_BlockGS, self).__init__(controller)

    def store_values(self, MS):
        """
        No overhead means nothing to store!
        """
        pass

    def extrapolation_estimate(self, MS):
        """
        The extrapolation estimate combines values of u and f from multiple steps to extrapolate and compare to the
        solution obtained by the time marching scheme.
        """

        # this is needed since we don't store anything
        self.dt = np.array([S.dt for S in MS])
        self.t = np.array([S.time for S in MS]) + self.dt

        if len(MS) > self.n:
            if None in self.u_coeff or self.params.use_adaptivity:
                self.get_extrapolation_coefficients()

            if len(MS[-1].levels) > 1:
                raise NotImplementedError('Extrapolated estimate only works on the finest level for now')

            # loop to go through all steps which we can extrapolate to
            for j in range(self.n, len(MS)):
                u_ex = MS[-1].levels[0].u[-1] * 0.

                # loop to sum up contributions from previous steps
                for i in range(1, self.n + 1):
                    L = MS[j - i].levels[0]
                    if type(L.f[-1]) == imex_mesh:
                        u_ex += self.u_coeff[-i] * L.u[-1] + self.f_coeff[-i] * (L.f[-1].impl + L.f[-1].expl)
                    elif type(L.f[-1]) == mesh:
                        u_ex += self.u_coeff[-i] * L.u[-1] + self.f_coeff[-i] * L.f[-1]
                    else:
                        raise DataError(f'Datatype {type(L.f[-1])} not supported by parallel extrapolation error estim\
ate!')
                MS[j].levels[0].status.error_extrapolation_estimate = abs(u_ex - MS[j].levels[0].u[-1]) * self.prefactor

    def estimate(self, MS):
        # loop in reverse through the block since later steps lag behind with iterations
        for i in range(len(MS) - 1, -1, -1):
            S = MS[i]
            if self.params.use_HotRod:
                if S.status.iter == S.params.maxiter - 1:
                    self.extrapolation_estimate(MS[:i + 1])
                elif S.status.iter == S.params.maxiter:
                    self.embedded_estimate_local_error(MS[:i + 1])
                    break

            else:
                # only estimate errors when last sweep is performed and not when doing Hot Rod
                if S.status.iter == S.params.maxiter:

                    if self.params.use_extrapolation_estimate:
                        self.extrapolation_estimate(MS[:i + 1])

                    if self.params.use_embedded_estimate or self.params.use_adaptivity:
                        self.embedded_estimate_local_error(MS[:i + 1])


def get_ErrorEstimator_nonMPI(controller):
    """

    This function should be called from the controller and return the correct version of the error estimator based on
    the chosen parameters.

    """

    if len(controller.MS) >= (controller.MS[0].params.maxiter + 4) // 2:
        return _ErrorEstimator_nonMPI_no_memory_overhead_BlockGS(controller)
    else:
        return _ErrorEstimator_nonMPI_BlockGS(controller)
