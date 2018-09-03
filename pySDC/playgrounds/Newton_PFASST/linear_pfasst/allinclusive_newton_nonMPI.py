import numpy as np
import itertools
import time

from pySDC.core.Errors import ControllerError
from pySDC.implementations.controller_classes.allinclusive_multigrid_nonMPI import allinclusive_multigrid_nonMPI


class allinclusive_newton_nonMPI(allinclusive_multigrid_nonMPI):
    """

    PFASST controller, running serialized version of PFASST in blocks (MG-style), linear only

    """

    # TODO: fix treatment of active slots in compute_rhs and set_jacobian
    # TODO: fix output: don't do print statement, use hooks (new ones?)
    # TODO: allow change of norm_res tolerance via parameter (!!!!!!!!)
    # TODO: move set_jacobian into restart_block_linear

    def __init__(self, num_procs, controller_params, description):
        """
        Initialization routine for PFASST controller
        """

        # call parent's initialization routine
        super(allinclusive_newton_nonMPI, self).__init__(num_procs, controller_params, description)

        self.ninnersolve = 0
        self.nouteriter = 0

    def run_with_pred(self, uk, u0, t0, Tend):

        # some initializations and reset of statistics
        uend = None
        num_procs = len(self.MS)
        self.hooks.reset_stats()

        # initialize time variables of each step

        # initial ordering of the steps: 0,1,...,Np-1
        slots = [p for p in range(num_procs)]

        # initialize time variables of each step
        time = [t0 + sum(self.MS[j].dt for j in range(p)) for p in slots]

        # determine which steps are still active (time < Tend)
        active = [time[p] < Tend - 10 * np.finfo(float).eps for p in slots]

        # compress slots according to active steps, i.e. remove all steps which have times above Tend
        active_slots = list(itertools.compress(slots, active))

        P = self.MS[0].levels[0].prob
        einit = [[P.dtype_u(P.init, val=0.0) for _ in self.MS[l].levels[0].u[1:]] for l in active_slots]

        # call pre-run hook
        for S in self.MS:
            self.hooks.pre_run(step=S, level_number=0)

        rhs, norm_rhs = self.compute_rhs(uk=uk, u0=u0, time=time)
        self.set_jacobian(uk=uk)

        print(norm_rhs)

        # main loop: as long as at least one step is still active (time < Tend), do something
        while any(active):

            k = 0
            ninnersolve = 0
            while norm_rhs > 1E-08:  # TODO!!!
                k += 1

                # initialize block of steps with u0
                self.restart_block_linear(active_slots, time, rhs, einit)

                MS_active = [self.MS[p] for p in active_slots]

                while not all([S.status.done for S in MS_active]):
                    MS_active = self.pfasst(MS_active)

                for p in range(len(MS_active)):
                    self.MS[active_slots[p]] = MS_active[p]

                    # uend is uend of the last active step in the list
                ek = [[P.dtype_u(self.MS[l].levels[0].u[m + 1]) for m in range(len(uk[l]))] for l in active_slots]
                uk = [[P.dtype_u(uk[l][m] - ek[l][m]) for m in range(len(uk[l]))] for l in active_slots]

                rhs, norm_rhs = self.compute_rhs(uk=uk, u0=u0, time=time)
                self.set_jacobian(uk=uk)

                ninnersolve = sum([self.MS[l].levels[0].prob.inner_solve_counter for l in active_slots])
                print('  Outer Iteration: %i -- number of inner solves: %i -- Newton residual: %8.6e'
                      % (k, ninnersolve, norm_rhs))

            for p in active_slots:
                time[p] += num_procs * self.MS[p].dt

            self.nouteriter += k
            self.ninnersolve += ninnersolve

            # determine new set of active steps and compress slots accordingly
            active = [time[p] < Tend - 10 * np.finfo(float).eps for p in slots]
            active_slots = list(itertools.compress(slots, active))

        # call post-run hook
        for S in self.MS:
            self.hooks.post_run(step=S, level_number=0)

        return uend, self.hooks.return_stats()

    def run(self, u0, t0, Tend):

        # some initializations and reset of statistics
        uend = None
        num_procs = len(self.MS)
        self.hooks.reset_stats()

        # initialize time variables of each step

        # initial ordering of the steps: 0,1,...,Np-1
        slots = [p for p in range(num_procs)]

        # initialize time variables of each step
        time = [t0 + sum(self.MS[j].dt for j in range(p)) for p in slots]

        # determine which steps are still active (time < Tend)
        active = [time[p] < Tend - 10 * np.finfo(float).eps for p in slots]

        # compress slots according to active steps, i.e. remove all steps which have times above Tend
        active_slots = list(itertools.compress(slots, active))

        P = self.MS[0].levels[0].prob
        uk = [[P.dtype_u(u0) for _ in self.MS[l].levels[0].u[1:]] for l in active_slots]
        einit = [[P.dtype_u(P.init, val=0.0) for _ in self.MS[l].levels[0].u[1:]] for l in active_slots]

        # call pre-run hook
        for S in self.MS:
            self.hooks.pre_run(step=S, level_number=0)

        rhs, norm_rhs = self.compute_rhs(uk=uk, u0=u0, time=time)
        self.set_jacobian(uk=uk)

        print(norm_rhs)

        # main loop: as long as at least one step is still active (time < Tend), do something
        while any(active):

            k = 0
            ninnersolve = 0
            while norm_rhs > 1E-08:  # TODO!!!
                k += 1

                # initialize block of steps with u0
                self.restart_block_linear(active_slots, time, rhs, einit)

                MS_active = [self.MS[p] for p in active_slots]

                while not all([S.status.done for S in MS_active]):
                    MS_active = self.pfasst(MS_active)

                for p in range(len(MS_active)):
                    self.MS[active_slots[p]] = MS_active[p]

                    # uend is uend of the last active step in the list
                ek = [[P.dtype_u(self.MS[l].levels[0].u[m + 1]) for m in range(len(uk[l]))] for l in active_slots]
                uk = [[P.dtype_u(uk[l][m] - ek[l][m]) for m in range(len(uk[l]))] for l in active_slots]

                rhs, norm_rhs = self.compute_rhs(uk=uk, u0=u0, time=time)
                self.set_jacobian(uk=uk)

                ninnersolve = sum([self.MS[l].levels[0].prob.inner_solve_counter for l in active_slots])
                print('  Outer Iteration: %i -- number of inner solves: %i -- Newton residual: %8.6e'
                      % (k, ninnersolve, norm_rhs))

            for p in active_slots:
                time[p] += num_procs * self.MS[p].dt

            self.nouteriter += k
            self.ninnersolve += ninnersolve

            # determine new set of active steps and compress slots accordingly
            active = [time[p] < Tend - 10 * np.finfo(float).eps for p in slots]
            active_slots = list(itertools.compress(slots, active))

        # call post-run hook
        for S in self.MS:
            self.hooks.post_run(step=S, level_number=0)

        uend = uk[-1][-1]

        return uend, self.hooks.return_stats()

    def restart_block_linear(self, active_slots, time, rhs, uk0):
        """
        Helper routine to reset/restart block of (active) steps

        """

        for j in range(len(active_slots)):

            # get slot number
            p = active_slots[j]

            # store current slot number for diagnostics
            self.MS[p].status.slot = p
            # store link to previous step
            self.MS[p].prev = self.MS[active_slots[j - 1]]
            # resets step
            self.MS[p].reset_step()
            # determine whether I am the first and/or last in line
            self.MS[p].status.first = active_slots.index(p) == 0
            self.MS[p].status.last = active_slots.index(p) == len(active_slots) - 1
            # initialize step with
            self.MS[p].init_step(uk0[p])
            u0 = self.MS[p].levels[0].prob.dtype_u(self.MS[p].levels[0].prob.init, val=0.0)
            self.MS[p].init_step(u0)
            self.MS[p].levels[0].sweep.compute_end_point()
            # reset some values
            self.MS[p].status.done = False
            self.MS[p].status.iter = 0
            self.MS[p].status.stage = 'SPREAD'
            for k in self.MS[p].levels:
                k.tag = None
                k.status.sweep = 1
                k.status.time = time[p]

            for m in range(len(rhs[p])):
                self.MS[p].levels[0].rhs[m] = self.MS[p].levels[0].prob.dtype_f(rhs[p][m])

    def compute_rhs(self, uk=None, u0=None, time=None):

        rhs = [[self.MS[0].levels[0].prob.dtype_u(u) for u in ustep] for ustep in uk]

        norm_rhs = 0.0
        for l, S in enumerate(self.MS):

            L = S.levels[0]
            P = L.prob

            f_ode = []
            for m in range(len(rhs[l])):
                f_ode.append(P.eval_f_ode(u=uk[l][m], t=time[l] + L.sweep.coll.nodes[m]))

            for m in range(len(rhs[l])):
                int = P.dtype_u(P.init, val=0.0)
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

        for j, S in enumerate(self.MS):
            S.levels[0].prob.build_jacobian(u=uk[-1][-1])

            for l in range(len(S.levels) - 1):

                base_transfer = S.get_transfer_class(S.levels[l], S.levels[l + 1])
                uc = base_transfer.space_transfer.restrict(uk[-1][-1])
                S.levels[l + 1].prob.build_jacobian(u=uc)

                # Tcf = base_transfer.space_transfer.Pspace
                # Tfc = base_transfer.space_transfer.Rspace
                #
                # S.levels[l + 1].prob.Jf = Tfc.dot(S.levels[l].prob.Jf).dot(Tcf)
