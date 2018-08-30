import itertools
import copy as cp
import numpy as np
import dill

from pySDC.core.Controller import controller
from pySDC.core.Errors import ControllerError, CommunicationError
from pySDC.implementations.controller_classes.allinclusive_multigrid_nonMPI import allinclusive_multigrid_nonMPI


class allinclusive_linearmultigrid_nonMPI(allinclusive_multigrid_nonMPI):
    """

    PFASST controller, running serialized version of PFASST in blocks (MG-style), linear only

    """

    def __init__(self, num_procs, controller_params, description):
        """
        Initialization routine for PFASST controller
        """

        # call parent's initialization routine
        super(allinclusive_linearmultigrid_nonMPI, self).__init__(num_procs, controller_params, description)

    def run(self, rhs, t0, Tend):
        raise ControllerError('This run routine does not exist, use run_linear instead')

    def run_linear(self, rhs, uk0, t0, Tend):
        """
        Main driver for running the serial version of SDC, MSSDC, MLSDC and PFASST (virtual parallelism)
        """

        # some initializations and reset of statistics

        num_procs = len(self.MS)
        self.hooks.reset_stats()

        assert len(rhs) == num_procs, \
            'ERROR: rhs needs to be a list of %i lists of entries for the RHS of the problem, got %s' % rhs

        # initialize time variables of each step
        time = [t0 + sum(S.dt for S in self.MS[:p]) for p in range(len(self.MS))]

        assert np.isclose(time[-1] + self.MS[-1].dt, Tend), \
            'ERROR: this controller can only run a single but full block'

        # initialize block of steps with uk0
        self.restart_block(time, rhs, uk0)

        # call pre-run hook
        for S in self.MS:
            self.hooks.pre_run(step=S, level_number=0)

        # main loop: as long as at least one step is still active (time < Tend), do something
        while not all([S.status.done for S in self.MS]):
            self.MS = self.pfasst(self.MS)

        uend = [[S.levels[0].u[m] for m in range(1, S.levels[0].sweep.coll.num_nodes + 1)] for S in self.MS]

        # call post-run hook
        for S in self.MS:
            self.hooks.post_run(step=S, level_number=0)

        return uend, self.hooks.return_stats()

    def restart_block(self, time, rhs, uk0):
        """
        Helper routine to reset/restart block of (active) steps

        """

        for l, S in enumerate(self.MS):

            # store current slot number for diagnostics
            S.status.slot = l
            # store link to previous step
            S.prev = self.MS[l - 1]
            # resets step
            S.reset_step()
            # determine whether I am the first and/or last in line
            S.status.first = l == 0
            S.status.last = l == len(self.MS) - 1
            # initialize step with
            S.init_step(uk0[l])
            u0 = S.levels[0].prob.dtype_u(S.levels[0].prob.init, val=0.0)
            S.init_step(u0)
            S.levels[0].sweep.compute_end_point()
            # reset some values
            S.status.done = False
            S.status.iter = 0
            S.status.stage = 'SPREAD'
            for k in S.levels:
                k.tag = None
                k.status.sweep = 1
                k.status.time = time[l]  # TODO: is this ok?

            for m in range(len(rhs[l])):
                S.levels[0].rhs[m] = S.levels[0].prob.dtype_f(rhs[l][m])

    def compute_rhs(self, uk=None, u0=None, t0=None):

        time = [t0 + sum(S.dt for S in self.MS[:p]) for p in range(len(self.MS))]

        rhs = [[u for u in ustep] for ustep in uk]

        norm_rhs = 0.0
        for l, S in enumerate(self.MS):

            L = S.levels[0]
            P = L.prob

            f_ode = []
            for m in range(len(rhs[l])):
                f_ode.append(P.eval_f_ode(u=uk[l][m], t=time[l]+L.sweep.coll.nodes[m]))

            for m in range(len(rhs[l])):
                int = P.dtype_u(P.init, val=0)
                for j in range(len(rhs[l])):
                    int += L.dt * L.sweep.coll.Qmat[m + 1, j + 1] * f_ode[j]
                rhs[l][m] -= int
                # This is where we need to communicate!
                if l > 0:
                    rhs[l][m] -= uk[l-1][-1]
                else:
                    rhs[l][m] -= u0

                norm_rhs = max(norm_rhs, abs(rhs[l][m]))

        return rhs, norm_rhs

    def set_jacobian(self, uk=None):

        # WARNING: this is simplified Newton!
        # The problem class does not know of different nodes besides the actual time, so we cannot have different Jf!
        for S in self.MS:
            S.levels[0].prob.build_jacobian(u=uk[-1][-1])

            for l in range(len(S.levels) - 1):

                base_transfer = S.get_transfer_class(S.levels[l], S.levels[l + 1])

                Tcf = base_transfer.space_transfer.Pspace
                Tfc = base_transfer.space_transfer.Rspace

                S.levels[l + 1].prob.Jf = Tfc.dot(S.levels[l].prob.Jf).dot(Tcf)
