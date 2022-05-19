import numpy as np
from scipy.special import factorial

from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh
from pySDC.core.Errors import DataError


class ErrorEstimator:

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

    def get_extrapolation_coefficients(self):
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

        # construct A matrix
        A = np.zeros((self.order, self.order))
        A[0, 0:self.n] = 1.
        j = np.arange(self.order)
        inv_facs = 1. / factorial(j)
        idx = np.argsort(t)
        steps_from_now = -np.cumsum([dt[idx[i]] for i in range(len(t))][::-1])[::-1]
        for i in range(1, self.order):
            A[i, :self.n] = steps_from_now**j[i] * inv_facs[i]
            A[i, self.n:self.order] = steps_from_now[2 * self.n - self.order:]**(j[i] - 1) * inv_facs[i - 1]

        # prepare rhs
        b = np.zeros(self.order)
        b[0] = 1.

        # solve linear system for the coefficients
        coeff = np.linalg.solve(A, b)
        self.u_coeff = coeff[:self.n]
        self.f_coeff[self.n * 2 - self.order:] = coeff[self.n:self.order]

        # determine prefactor
        r = abs(self.dt / self.dt[-1])**(self.order - 1)
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
            self.t[oldest_val] = S.levels[0].time + S.levels[0].dt
            self.dt[oldest_val] = S.levels[0].dt

    def embedded_estimate(self, S):
        """
        Compute embedded error estimate on the last node of each level
        """
        for L in S.levels:
            # order rises by one between sweeps, making this so ridiculously easy
            L.status.error_embedded_estimate = abs(L.uold[-1] - L.u[-1])

    def extrapolation_estimate(self, S):
        """
        The extrapolation estimate combines values of u and f from multiple steps to extrapolate and compare to the
        solution obtained by the time marching scheme.
        """
        if None not in self.dt:
            if None in self.u_coeff or self.params.use_adaptivity:
                self.get_extrapolation_coefficients()

            if len(S.levels) > 1:
                raise NotImplementedError('Extrapolated estimate only works on the finest level for now')

            u_ex = S.levels[0].u[-1] * 0.
            idx = np.argsort(self.t)
            for i in range(self.n):
                u_ex += self.u_coeff[i] * self.u[idx[i]] + self.f_coeff[i] * self.f[idx[i]]

            S.levels[0].status.error_extrapolation_estimate = \
                abs(u_ex - S.levels[0].u[-1]) * self.prefactor

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


class ErrorEstimator_nonMPI(ErrorEstimator):

    def __init__(self, controller):
        super(ErrorEstimator_nonMPI, self).__init__(controller, order=controller.MS[0].params.maxiter,
                                                    size=len(controller.MS))

        if self.params.use_extrapolation_estimate:
            if not controller.MS[0].levels[0].params.restol == 0:
                raise NotImplementedError('Extrapolation based error estimate so far only with fixed order')
            maxiter = [0] * len(controller.MS)
            for i in range(len(controller.MS)):
                maxiter[i] = controller.MS[i].params.maxiter
            if not maxiter.count(maxiter[0]) == len(maxiter):
                raise NotImplementedError('All steps need to have the same order in time so far')

    def store_values(self, MS):
        for S in MS:
            super(ErrorEstimator_nonMPI, self).store_values(S)

    def communicate_time(self):
        return self.t, self.dt

    def estimate(self, MS):
        for S in MS:
            super(ErrorEstimator_nonMPI, self).estimate(S)

    def setup_extrapolation(self, controller, order, size):
        super(ErrorEstimator_nonMPI, self).setup_extrapolation(controller, order, size)
        # create variables to store u, f, t and dt from previous steps
        self.u = [None] * self.n_per_proc * size
        self.f = [None] * self.n_per_proc * size
        self.t = np.array([None] * self.n_per_proc * size)
        self.dt = np.array([None] * self.n_per_proc * size)
