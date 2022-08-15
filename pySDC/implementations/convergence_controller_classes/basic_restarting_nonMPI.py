from pySDC.core.ConvergenceController import ConvergenceController


class BasicRestartingNonMPI(ConvergenceController):
    '''
    Class with some utilities for restarting. The specific functions are:
     - Telling each step after one that requested a restart to get restarted as well
     - Allowing each step to be restarted a limited number of times in a row before just moving on anyways

    Default control order is 100.
    '''

    def setup(self, controller, params, description):
        '''
        Define parameters here

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        '''
        defaults = {
            'control_order': 95,
            'max_restarts': 1 if len(controller.MS) == 1 else 2,
        }

        return {**defaults, **params}

    def determine_restart(self, controller, S):
        '''
        Restart all steps after the first one which wants to be restarted as well, but also check if we lost patience
        with the restarts and want to move on anyways.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        '''
        # check if we performed too many restarts
        if S.status.first:
            self.max_restart_reached = S.status.restarts_in_a_row >= self.params.max_restarts
            if self.max_restart_reached and S.status.restart:
                self.log(f'Step(s) restarted {S.status.restarts_in_a_row} time(s) already, maximum reached, moving \
on...', S)

        self.restart = S.status.restart or self.restart
        S.status.restart = (S.status.restart or self.restart) and not self.max_restart_reached

        return None

    def reset_global_variables_nonMPI(self, controller):
        '''
        Reset all variables with are used to simulate communication here

        Args:
            controller (pySDC.Controller): The controller

        Returns:
            None
        '''
        self.restart = False
        self.max_restart_reached = False

        return None

    def prepare_next_block_nonMPI(self, controller, MS, active_slots, time, Tend):
        """
        Update restarts in a row for all steps.

        Args:
            controller (pySDC.Controller): The controller
            MS (list): List of the steps of the controller
            active_slots (list): Index list of active steps
            time (list): List containing the time of all the steps
            Tend (float): Final time of the simulation

        Returns:
            None
        """
        for p in active_slots:
            MS[p].status.restarts_in_a_row = MS[p].status.restarts_in_a_row + 1 if MS[p].status.restart else 0

        return None
