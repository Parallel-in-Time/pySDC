import itertools
import copy as cp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pySDC.implementations.controller_classes.allinclusive_multigrid_nonMPI import allinclusive_multigrid_nonMPI
from pySDC.core.Errors import ControllerError, CommunicationError

from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit


class allinclusive_matrix_nonMPI(allinclusive_multigrid_nonMPI):
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

        assert description['dtype_u'] is mesh, \
            'ERROR: matrix version will only work with mesh data type for u, got %s' % description['dtype_u']
        assert description['dtype_f'] is mesh, \
            'ERROR: matrix version will only work with mesh data type for f, got %s' % description['dtype_f']
        assert description['sweeper_class'] is generic_implicit, \
            'ERROR: matrix version will only work with generic_implicit sweeper, got %s' % description['sweeper_class']

        # call parent's initialization routine
        super(allinclusive_matrix_nonMPI, self).__init__(num_procs=num_procs, controller_params=controller_params,
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
        assert [level.sweep.coll.num_nodes for step in self.MS for level in step.levels].count(self.nnodes) == \
            self.nlevels * self.nsteps, 'ERROR: nnodes must be equal for all steps and all levels'
        assert [type(level.prob) for step in self.MS for level in step.levels].count(type(prob)) == \
            self.nlevels * self.nsteps, 'ERROR: all probem classes have to be the same'

        assert self.params.predict is False, 'ERROR: no predictor for matrix controller yet'  # TODO: fixme

        assert hasattr(prob, 'A'), 'ERROR: need system matrix A for this (and linear problems!)'

        A = prob.A
        Q = self.MS[0].levels[0].sweep.coll.Qmat[1:, 1:]
        Qd = self.MS[0].levels[0].sweep.QI[1:, 1:]

        E = np.zeros((self.nsteps, self.nsteps))
        np.fill_diagonal(E[1:, :], 1)

        N = np.zeros((self.nnodes, self.nnodes))
        N[:, -1] = 1

        self.C = sp.eye(self.nsteps * self.nnodes * self.nspace) - \
            self.dt * sp.kron(sp.eye(self.nsteps), sp.kron(Q, A)) - sp.kron(E, sp.kron(N, sp.eye(self.nspace)))
        self.P = sp.eye(self.nsteps * self.nnodes * self.nspace) - \
            self.dt * sp.kron(sp.eye(self.nsteps), sp.kron(Qd, A))

        if self.nlevels > 1:
            prob_c = self.MS[0].levels[1].prob
            self.nspace_c = prob_c.init

            Ac = prob_c.A
            Qdc = self.MS[0].levels[1].sweep.QI[1:, 1:]

            TcfA = self.MS[0].base_transfer.space_transfer.Pspace
            TfcA = self.MS[0].base_transfer.space_transfer.Rspace

            self.Tcf = sp.kron(sp.eye(self.nsteps * self.nnodes), TcfA)
            self.Tfc = sp.kron(sp.eye(self.nsteps * self.nnodes), TfcA)

            self.Pc = sp.eye(self.nsteps * self.nnodes * self.nspace_c) - \
                self.dt * sp.kron(sp.eye(self.nsteps), sp.kron(Qdc, Ac)) - sp.kron(E, sp.kron(N, sp.eye(self.nspace_c)))

        self.u = np.zeros(self.nsteps * self.nnodes * self.nspace)
        self.res = np.zeros(self.nsteps * self.nnodes * self.nspace)
        self.u0 = np.zeros(self.nsteps * self.nnodes * self.nspace)

    def run(self, u0, t0, Tend):
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
        uend = None
        num_procs = len(self.MS)
        self.hooks.reset_stats()

        assert ((Tend - t0) / self.dt).is_integer(), \
            'ERROR: dt, t0, Tend were not chosen correctly, do not divide interval to be computed equally'

        assert int((Tend - t0) / self.dt) % num_procs == 0, 'ERROR: num_procs not chosen correctly'

        # initial ordering of the steps: 0,1,...,Np-1
        slots = [p for p in range(num_procs)]

        # initialize time variables of each step
        time = [t0 + sum(self.dt for _ in range(p)) for p in slots]

        # initialize block of steps with u0
        self.restart_block(slots, time, u0)

        # call pre-run hook
        for S in self.MS:
            self.hooks.pre_run(step=S, level_number=0)

        nblocks = int((Tend - t0) / self.dt / num_procs)

        for i in range(nblocks):

            self.MS = self.pfasst(self.MS)

            for p in slots:
                time[p] += num_procs * self.dt

            # uend is uend of the last active step in the list
            uend = self.MS[-1].levels[0].uend
            self.restart_block(slots, time, uend)

        # call post-run hook
        for S in self.MS:
            self.hooks.post_run(step=S, level_number=0)

        return uend, self.hooks.return_stats()

    def restart_block(self, slots, time, u0):
        """
        Helper routine to reset/restart block of steps

        Args:
            slots: list of steps
            time: list of new times
            u0: initial value to distribute across the steps

        """

        # loop over steps
        for p in slots:

            # store current slot number for diagnostics
            self.MS[p].status.slot = p

        for p in slots:
            for lvl in self.MS[p].levels:
                lvl.status.time = time[p]
                P = lvl.prob
                for m in range(1, self.nnodes + 1):
                    lvl.u[m] = P.dtype_u(init=P.init, val=0)
                    lvl.f[m] = P.dtype_f(init=P.init, val=0)

        self.u0 = np.kron(np.concatenate([[1], [0] * (self.nsteps - 1)]), np.kron(np.ones(self.nnodes), u0.values))

        if self.MS[0].levels[0].sweep.params.spread:
            self.u = np.kron(np.ones(self.nsteps * self.nnodes), u0.values)
        else:
            self.u = self.u0.copy()

        self.res = np.zeros(self.nsteps * self.nnodes * self.nspace)

    @staticmethod
    def update_data(MS, u, res, niter, level, stage):

        for S in MS:
            S.status.stage = stage
            S.status.iter = niter

            L = S.levels[level]
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

        self.res = self.u0 - self.C.dot(self.u)

        MS = self.update_data(MS=MS, u=self.u, res=self.res, niter=niter, level=0, stage='PRE_STEP')
        for S in MS:
            self.hooks.pre_step(step=S, level_number=0)

        for _ in range(MS[0].levels[0].params.nsweeps):

            MS = self.update_data(MS=MS, u=self.u, res=self.res, niter=niter, level=0, stage='PRE_PRE_SWEEP')
            for S in MS:
                self.hooks.pre_sweep(step=S, level_number=0)

            self.u += spla.spsolve(self.P, self.res)
            self.res = self.u0 - self.C.dot(self.u)

            MS = self.update_data(MS=MS, u=self.u, res=self.res, niter=niter, level=0, stage='POST_PRE_SWEEP')
            for S in MS:
                self.hooks.post_sweep(step=S, level_number=0)

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

                    self.u += self.Tcf.dot(spla.spsolve(self.Pc, self.Tfc.dot(self.res)))
                    self.res = self.u0 - self.C.dot(self.u)

                    MS = self.update_data(MS=MS, u=self.u, res=self.res, niter=niter, level=1,
                                          stage='POST_COARSE_SWEEP')
                    for S in MS:
                        self.hooks.post_sweep(step=S, level_number=1)

            for _ in range(MS[0].levels[0].params.nsweeps):

                MS = self.update_data(MS=MS, u=self.u, res=self.res, niter=niter, level=0, stage='PRE_FINE_SWEEP')
                for S in MS:
                    self.hooks.pre_sweep(step=S, level_number=0)

                self.u += spla.spsolve(self.P, self.res)
                self.res = self.u0 - self.C.dot(self.u)

                MS = self.update_data(MS=MS, u=self.u, res=self.res, niter=niter, level=0, stage='POST_FINE_SWEEP')
                for S in MS:
                    self.hooks.post_sweep(step=S, level_number=0)

            MS = self.update_data(MS=MS, u=self.u, res=self.res, niter=niter, level=0, stage='POST_ITERATION')
            for S in MS:
                self.hooks.post_iteration(step=S, level_number=0)

        MS = self.update_data(MS=MS, u=self.u, res=self.res, niter=niter, level=0, stage='POST_STEP')
        for S in MS:
            self.hooks.post_step(step=S, level_number=0)

        return MS
