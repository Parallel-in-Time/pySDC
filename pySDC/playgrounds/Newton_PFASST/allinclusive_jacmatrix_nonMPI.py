import numpy as np
import scipy.sparse as spla

from pySDC.implementations.controller_classes.allinclusive_multigrid_nonMPI import allinclusive_multigrid_nonMPI

from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.implementations.datatype_classes.complex_mesh import mesh as cmesh
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit


class allinclusive_jacmatrix_nonMPI(allinclusive_multigrid_nonMPI):
    """

    PFASST controller, running serial matrix-based versions

    """

    def __init__(self, num_procs, controller_params, description):
        """
       Initialization routine for PFASST controller

       Args:
           num_procs: number of parallel time steps (still serial, though), can be 1
           controller_params: parameter set for the controller and the steps
           description: all the parameters to set up the rest (levels, problems, transfer, ...)
       """

        assert description['dtype_u'] is mesh or cmesh, \
            'ERROR: matrix version will only work with mesh data type for u, got %s' % description['dtype_u']
        assert description['dtype_f'] is mesh or cmesh, \
            'ERROR: matrix version will only work with mesh data type for f, got %s' % description['dtype_f']
        assert description['sweeper_class'] is generic_implicit, \
            'ERROR: matrix version will only work with generic_implicit sweeper, got %s' % description['sweeper_class']

        # call parent's initialization routine
        super(allinclusive_jacmatrix_nonMPI, self).__init__(num_procs=num_procs, controller_params=controller_params,
                                                         description=description)

        self.nsteps = len(self.MS)
        self.nlevels = len(self.MS[0].levels)
        self.nnodes = self.MS[0].levels[0].sweep.coll.num_nodes
        self.nspace = self.MS[0].levels[0].prob.init

        self.dt = self.MS[0].levels[0].dt
        self.tol = self.MS[0].levels[0].params.restol
        self.maxiter = self.MS[0].params.maxiter

        prob = self.MS[0].levels[0].prob

        assert isinstance(self.nspace, int), 'ERROR: can only handle 1D data, got %s' % self.nspace
        assert [type(level.sweep.coll) for step in self.MS for level in step.levels].count(CollGaussRadau_Right) == \
            self.nlevels * self.nsteps, 'ERROR: all collocation nodes have to be of Gauss-Radau type'
        assert self.nlevels <= 2, 'ERROR: cannot use matrix-PFASST with more than 2 levels'  # TODO: fixme
        assert [level.dt for step in self.MS for level in step.levels].count(self.dt) == self.nlevels * self.nsteps, \
            'ERROR: dt must be equal for all steps and all levels'
        # assert [level.sweep.coll.num_nodes for step in self.MS for level in step.levels].count(self.nnodes) == \
        #     self.nlevels * self.nsteps, 'ERROR: nnodes must be equal for all steps and all levels'
        assert [type(level.prob) for step in self.MS for level in step.levels].count(type(prob)) == \
            self.nlevels * self.nsteps, 'ERROR: all probem classes have to be the same'

        assert self.params.predict is False, 'ERROR: no predictor for matrix controller yet'  # TODO: fixme

        assert hasattr(prob, 'A'), 'ERROR: need system matrix A for this (and linear problems!)'

        # initial ordering of the steps: 0,1,...,Np-1
        slots = [p for p in range(num_procs)]

        # loop over steps
        for p in slots:
            # store current slot number for diagnostics
            self.MS[p].status.slot = p

        self.u = np.zeros(self.nsteps * self.nnodes * self.nspace)
        self.res = np.zeros(self.nsteps * self.nnodes * self.nspace)
        self.rhs = np.zeros(self.nsteps * self.nnodes * self.nspace)

        self.iter_counter = 0

    def compute_rhs(self, uk, t0):

        u0 = self.MS[0].levels[0].prob.u_exact(t0)

        u0_full = np.kron(np.concatenate([[1], [0] * (self.nsteps - 1)]), np.kron(np.ones(self.nnodes), u0.values))

        self.rhs = uk - u0_full

        E = np.zeros((self.nsteps, self.nsteps))
        np.fill_diagonal(E[1:, :], 1)

        N = np.zeros((self.nnodes, self.nnodes))
        N[:, -1] = 1

        self.rhs -= np.kron(E, np.kron(N, np.eye(self.nspace))).dot(uk)

        for S in self.MS:
            L = S.levels[0]
            P = L.prob
            first = S.status.slot * self.nnodes * self.nspace
            last = (S.status.slot + 1) * self.nnodes * self.nspace
            for m in range(1, self.nnodes + 1):
                time = t0 + sum(self.dt for _ in range(S.status.slot))
                mstart = first + (m - 1) * self.nspace
                mend = first + m * self.nspace
                L.u[m] = P.dtype_u(init=P.init, val=0)
                L.u[m].values = uk[mstart:mend].copy()
                L.f[m] = P.dtype_f(init=P.init, val=0)
                L.f[m] = P.eval_f(L.u[m], time + L.dt * L.sweep.coll.nodes[m - 1])
            integral = L.sweep.integrate()
            self.rhs[first:last] -= np.concatenate([integral[m].values for m in range(self.nnodes)])

    def compute_matrices(self):

        L = self.MS[0].levels[0]
        P = L.prob

        jac = []
        for S in self.MS:
            L = S.levels[0]
            P = L.prob
            for m in range(1, self.nnodes + 1):
                jac.append(P.eval_jacobian(L.u[m]))
        A = spla.block_diag(jac).todense()
        Q = L.sweep.coll.Qmat[1:, 1:]
        Qd = L.sweep.QI[1:, 1:]

        E = np.zeros((self.nsteps, self.nsteps))
        np.fill_diagonal(E[1:, :], 1)

        N = np.zeros((self.nnodes, self.nnodes))
        N[:, -1] = 1
        self.C = np.eye(self.nsteps * self.nnodes * self.nspace) - \
                 self.dt * np.kron(np.eye(self.nsteps), np.kron(Q, np.eye(self.nspace))).dot(A) - np.kron(E, np.kron(N, np.eye(self.nspace)))
        self.C = np.array(self.C)
        self.P = np.eye(self.nsteps * self.nnodes * self.nspace) - \
                 self.dt * np.kron(np.eye(self.nsteps), np.kron(Qd, np.eye(self.nspace))).dot(A)
        self.P = np.array(self.P)

        if self.nlevels > 1:
            L = self.MS[0].levels[1]
            P = L.prob
            self.nspace_c = P.init

            Qdc = L.sweep.QI[1:, 1:]
            nnodesc = L.sweep.coll.num_nodes
            Nc = np.zeros((nnodesc, nnodesc))
            Nc[:, -1] = 1

            TcfA = self.MS[0].base_transfer.space_transfer.Pspace.todense()
            TfcA = self.MS[0].base_transfer.space_transfer.Rspace.todense()
            TcfQ = self.MS[0].base_transfer.Pcoll
            TfcQ = self.MS[0].base_transfer.Rcoll

            self.Tcf = np.array(np.kron(np.eye(self.nsteps), np.kron(TcfQ, TcfA)))
            self.Tfc = np.array(np.kron(np.eye(self.nsteps), np.kron(TfcQ, TfcA)))

            Ac = self.Tfc.dot(A.dot(self.Tcf))

            self.Pc = np.eye(self.nsteps * nnodesc * self.nspace_c) - \
                      self.dt * np.kron(np.eye(self.nsteps), np.kron(Qdc, np.eye(self.nspace_c))).dot(Ac) - \
                      np.kron(E, np.kron(Nc, np.eye(self.nspace_c)))
            self.Pc = np.array(self.Pc)

    def run(self, uk, t0, Tend):
        """
        Main driver for running the serial matrix version of SDC, MSSDC, MLSDC and PFASST

        Args:
           u0: initial values
           t0: starting time
           Tend: ending time

        Returns:
            end values on the finest level
            stats object containing statistics for each step, each level and each iteration
        """

        # some initializations and reset of statistics
        num_procs = len(self.MS)
        self.hooks.reset_stats()

        assert ((Tend - t0) / self.dt).is_integer(), \
            'ERROR: dt, t0, Tend were not chosen correctly, do not divide interval to be computed equally'

        assert int((Tend - t0) / self.dt) % num_procs == 0, 'ERROR: num_procs not chosen correctly'

        slots = [p for p in range(num_procs)]

        # initialize time variables of each step
        time = [t0 + sum(self.dt for _ in range(p)) for p in slots]

        for p in slots:
            for lvl in self.MS[p].levels:
                lvl.status.time = time[p]
                P = lvl.prob
                for m in range(1, lvl.sweep.coll.num_nodes + 1):
                    lvl.u[m] = P.dtype_u(init=P.init, val=0)
                    lvl.f[m] = P.dtype_f(init=P.init, val=0)

        self.compute_rhs(uk, t0)
        self.compute_matrices()

        self.u = np.zeros(self.nsteps * self.nnodes * self.nspace)

        # call pre-run hook
        for S in self.MS:
            self.hooks.pre_run(step=S, level_number=0)

        nblocks = int((Tend - t0) / self.dt / num_procs)

        assert nblocks == 1, 'ERROR: only one block of PFASST allowed'

        self.MS, self.u = self.pfasst(self.MS)

        # call post-run hook
        for S in self.MS:
            self.hooks.post_run(step=S, level_number=0)

        return self.u, self.hooks.return_stats()

    @staticmethod
    def update_data(MS, u, res, niter, level, stage):

        for S in MS:
            S.status.stage = stage
            S.status.iter = niter

            L = S.levels[level]
            L.status.sweep = 0

            P = L.prob
            nnodes = L.sweep.coll.num_nodes
            nspace = P.init

            first = S.status.slot * nnodes * nspace
            last = (S.status.slot + 1) * nnodes * nspace

            L.status.residual = np.linalg.norm(res[first:last], np.inf)

            for m in range(1, nnodes + 1):
                mstart = first + (m - 1) * nspace
                mend = first + m * nspace
                L.u[m].values = u[mstart:mend]
                L.f[m] = P.eval_f(L.u[m], L.time + L.dt * L.sweep.coll.nodes[m - 1])

            S.levels[level].sweep.compute_end_point()

        return MS

    def pfasst(self, MS):
        """
        Main function including the stages of SDC, MLSDC and PFASST (the "controller")

        Args:
            MS: all active steps

        Returns:
            all active steps
        """

        niter = 0

        self.res = self.rhs - self.C.dot(self.u)

        MS = self.update_data(MS=MS, u=self.u, res=self.res, niter=niter, level=0, stage='PRE_STEP')
        for S in MS:
            self.hooks.pre_step(step=S, level_number=0)

        while np.linalg.norm(self.res, np.inf) > self.tol and niter < self.maxiter:

            niter += 1

            MS = self.update_data(MS=MS, u=self.u, res=self.res, niter=niter, level=0, stage='PRE_ITERATION')
            for S in MS:
                self.hooks.pre_iteration(step=S, level_number=0)

            if self.nlevels > 1:
                for _ in range(MS[0].levels[1].params.nsweeps):

                    MS = self.update_data(MS=MS, u=self.u, res=self.res, niter=niter, level=1, stage='PRE_COARSE_SWEEP')
                    for S in MS:
                        self.hooks.pre_sweep(step=S, level_number=1)

                    self.u += self.Tcf.dot(np.linalg.solve(self.Pc, self.Tfc.dot(self.res)))
                    self.res = self.rhs - self.C.dot(self.u)

                    MS = self.update_data(MS=MS, u=self.u, res=self.res, niter=niter, level=1,
                                          stage='POST_COARSE_SWEEP')
                    for S in MS:
                        self.hooks.post_sweep(step=S, level_number=1)

            for _ in range(MS[0].levels[0].params.nsweeps):

                MS = self.update_data(MS=MS, u=self.u, res=self.res, niter=niter, level=0, stage='PRE_FINE_SWEEP')
                for S in MS:
                    self.hooks.pre_sweep(step=S, level_number=0)

                self.u += np.linalg.solve(self.P, self.res)
                self.res = self.rhs - self.C.dot(self.u)

                MS = self.update_data(MS=MS, u=self.u, res=self.res, niter=niter, level=0, stage='POST_FINE_SWEEP')
                for S in MS:
                    self.hooks.post_sweep(step=S, level_number=0)

            MS = self.update_data(MS=MS, u=self.u, res=self.res, niter=niter, level=0, stage='POST_ITERATION')
            for S in MS:
                self.hooks.post_iteration(step=S, level_number=0)

        MS = self.update_data(MS=MS, u=self.u, res=self.res, niter=niter, level=0, stage='POST_STEP')
        for S in MS:
            self.hooks.post_step(step=S, level_number=0)

        self.iter_counter += niter

        return MS, self.u
