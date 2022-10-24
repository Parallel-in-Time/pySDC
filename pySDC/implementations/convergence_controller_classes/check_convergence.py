import numpy as np

from pySDC.core.ConvergenceController import ConvergenceController


class CheckConvergence(ConvergenceController):
    """
    Perform simple checks on convergence for SDC iterations.

    Iteration is terminated via one of two criteria:
     - Residual tolerance
     - Maximum number of iterations
    """

    def setup(self, controller, params, description, **kwargs):
        """
        Define default parameters here

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        """
        return {"control_order": +200, **params}

    def check_iteration_status(self, controller, S, **kwargs):
        """
        Routine to determine whether to stop iterating (currently testing the residual + the max. number of iterations)

        Args:
            controller (pySDC.Controller.controller): The controller
            S (pySDC.Step.step): The current step

        Returns:
            None
        """

        # do all this on the finest level
        L = S.levels[0]
        L.sweep.compute_residual()

        # get residual and check against prescribed tolerance (plus check number of iterations
        res = L.status.residual
        converged = S.status.iter >= S.params.maxiter or res <= L.params.restol or S.status.force_done
        if converged is not None:
            S.status.done = converged

        if "comm" in kwargs.keys():
            self.communicate_convergence(controller, S, **kwargs)

        return None

    def communicate_convergence(self, controller, S, comm):
        from mpi4py import MPI

        """
        Communicate the convergence status

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step.step): The current step
            comm (mpi4py.MPI.Comm): MPI communicator
        """
        # Either gather information about all status or send forward own
        if controller.params.all_to_done:

            controller.hooks.pre_comm(step=S, level_number=0)
            S.status.done = comm.allreduce(sendobj=S.status.done, op=MPI.LAND)
            controller.hooks.post_comm(step=S, level_number=0, add_to_stats=True)

        else:

            controller.hooks.pre_comm(step=S, level_number=0)

            # check if an open request of the status send is pending
            controller.wait_with_interrupt(request=controller.req_status)
            if S.status.force_done:
                return None

            # recv status
            if not S.status.first and not S.status.prev_done:
                tmp = np.empty(1, dtype=int)
                comm.Irecv((tmp, MPI.INT), source=S.prev, tag=99).Wait()
                S.status.prev_done = tmp
                self.logger.debug(
                    "recv status: status %s, process %s, time %s, source %s, tag %s, iter %s"
                    % (
                        S.status.prev_done,
                        S.status.slot,
                        S.time,
                        S.prev,
                        99,
                        S.status.iter,
                    )
                )
                S.status.done = S.status.done and S.status.prev_done

            # send status forward
            if not S.status.last:
                self.logger.debug(
                    "isend status: status %s, process %s, time %s, target %s, tag %s, iter %s"
                    % (S.status.done, S.status.slot, S.time, S.next, 99, S.status.iter)
                )
                tmp = np.array(S.status.done, dtype=int)
                controller.req_status = comm.Issend((tmp, MPI.INT), dest=S.next, tag=99)

            controller.hooks.post_comm(step=S, level_number=0, add_to_stats=True)
