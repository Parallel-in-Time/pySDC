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
        defaults = {'control_order': +200, 'use_e_tol': 'e_tol' in description['level_params'].keys()}

        return {**defaults, **super().setup(controller, params, description, **kwargs)}

    def dependencies(self, controller, description, **kwargs):
        """
        Load the embedded error estimator if needed.

        Args:
            controller (pySDC.Controller): The controller
            description (dict): The description object used to instantiate the controller

        Returns:
            None
        """
        super().dependencies(controller, description)

        if self.params.use_e_tol:

            from pySDC.implementations.convergence_controller_classes.estimate_embedded_error import (
                EstimateEmbeddedError,
            )
            from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

            controller.add_convergence_controller(
                EstimateEmbeddedError.get_implementation("nonMPI" if type(controller) == controller_nonMPI else "MPI"),
                description=description,
            )

        return None

    @staticmethod
    def check_convergence(S):
        """
        Check the convergence of a single step.
        Test the residual and max. number of iterations as well as allowing overrides to both stop and continue.

        Args:
            S (pySDC.Step): The current step

        Returns:
            bool: Convergence status of the step
        """
        # do all this on the finest level
        L = S.levels[0]
        L.sweep.compute_residual()

        # get residual and check against prescribed tolerance (plus check number of iterations)
        iter_converged = S.status.iter >= S.params.maxiter
        res_converged = L.status.residual <= L.params.restol
        e_tol_converged = (
            L.status.error_embedded_estimate < L.params.e_tol
            if (L.params.get('e_tol') and L.status.get('error_embedded_estimate'))
            else False
        )
        converged = (
            iter_converged or res_converged or e_tol_converged or S.status.force_done
        ) and not S.status.force_continue
        if converged is None:
            converged = False
        return converged

    def check_iteration_status(self, controller, S, **kwargs):
        """
        Routine to determine whether to stop iterating (currently testing the residual + the max. number of iterations)

        Args:
            controller (pySDC.Controller.controller): The controller
            S (pySDC.Step.step): The current step

        Returns:
            None
        """
        S.status.done = self.check_convergence(S)

        if "comm" in kwargs.keys():
            self.communicate_convergence(controller, S, **kwargs)

        S.status.force_continue = False

        return None

    def communicate_convergence(self, controller, S, comm):
        """
        Communicate the convergence status during `check_iteration_status` if MPI is used.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step.step): The current step
            comm (mpi4py.MPI.Comm): MPI communicator

        Returns:
            None
        """
        # Either gather information about all status or send forward own
        if controller.params.all_to_done:
            from mpi4py.MPI import LAND

            for hook in controller.hooks:
                hook.pre_comm(step=S, level_number=0)
            S.status.done = comm.allreduce(sendobj=S.status.done, op=LAND)
            for hook in controller.hooks:
                hook.post_comm(step=S, level_number=0, add_to_stats=True)

        else:
            for hook in controller.hooks:
                hook.pre_comm(step=S, level_number=0)

            # check if an open request of the status send is pending
            controller.wait_with_interrupt(request=controller.req_status)
            if S.status.force_done:
                return None

            # recv status
            if not S.status.first and not S.status.prev_done:
                S.status.prev_done = self.recv(comm, source=S.status.slot - 1)
                S.status.done = S.status.done and S.status.prev_done

            # send status forward
            if not S.status.last:
                self.send(comm, dest=S.status.slot + 1, data=S.status.done)

            for hook in controller.hooks:
                hook.post_comm(step=S, level_number=0, add_to_stats=True)
