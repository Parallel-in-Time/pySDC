import logging
from pySDC.helpers.pysdc_helper import FrozenClass


# short helper class to add params as attributes
class _Pars(FrozenClass):
    def __init__(self, params):
        self.control_order = 0  # integer that determines the order in which the convergence controllers are called

        for k, v in params.items():
            setattr(self, k, v)

        self._freeze()


# short helper class to store status variables
class _Status(FrozenClass):
    '''
    Initialize status variables with None, since at the time of instantiation of the convergence controllers, not all
    relevant information about the controller are known.
    '''
    def __init__(self, status_variabes):

        [setattr(self, key, None) for key in status_variabes]

        self._freeze()


class ConvergenceController(object):
    """
    Base abstract class for convergence controller, which is plugged into the controller to determine the iteration
    count and time step size.
    """

    def __init__(self, controller, params, description):
        self.params = _Pars(self.setup(controller, params, description))
        params_ok, msg = self.check_parameters(controller, params, description)
        assert params_ok, msg
        self.dependencies(controller, description)
        self.logger = logging.getLogger(f'{type(self).__name__}')

    def log(self, msg, S, level=15):
        '''
        Shortcut that has a default level for the logger. 15 is above debug but below info.

        Args:
            msg (str): Meassage you want to log
            S (pySDC.step): The current step
            level (int): the level passed to the logger

        Returns:
            None
        '''
        self.logger.log(level, f'Process {S.status.slot:2d} on time {S.time:.6f} - {msg}')
        return None

    def setup(self, controller, params, description):
        '''
        Setup various variables that only need to be set once in the beginning.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary after setup
        '''
        return params

    def dependencies(self, controller, description):
        '''
        Load dependencies on other convergence controllers here.

        Args:
            controller (pySDC.Controller): The controller
            description (dict): The description object used to instantiate the controller

        Returns:
            None
        '''
        pass

    def check_parameters(self, controller, params, description):
        '''
        Check whether parameters are compatible with whatever assumptions went into the step size functions etc.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            bool: Whether the parameters are compatible
            str: The error message
        '''
        return True, ''

    def check_iteration_status(self, controller, S):
        '''
        Determine whether to keep iterating or not in this function.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        '''
        pass

    def get_new_step_size(self, controller, S):
        '''
        This function allows to set a step size with arbitrary criteria.
        Make sure to give an order to the convergence controller by setting the `control_order` variable in the params.
        This variable is an integer and you can see what the current order is by using
        `controller.print_convergence_controllers()`.

        Args:
            controller (pySDC.Controller.controller): The controller
            S (pySDC.step): The current step

        Returns:
            None
        '''
        pass

    def determine_restart(self, controller, S):
        '''
        Determine for each step separately if it wants to be restarted for whatever reason.

        Args:
            controller (pySDC.Controller.controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        '''
        pass

    def setup_status_variables(self, controller):
        '''
        Setup status variables.
        This is not done at the time of instatiation, since the controller is not fully instantiated at that time and
        hence not all information are available. Instead, this function is called after the controller has been fully
        instantiated.

        Args:
            controller (pySDC.Controller): The controller

        Reutrns:
            None
        '''
        return None

    def reset_buffers_nonMPI(self, controller):
        '''
        Buffers refer to variables used across multiple steps that are stored in the convergence controller classes to
        immitate communication in non mpi versions. These have to be reset in order to replicate avalability of
        variables in mpi versions.

        For instance, if step 0 sets self.buffers.x = 1 from self.buffers.x = 0, when the same MPI rank uses the
        variable with step 1, it will still carry the value of self.buffers.x = 1, equivalent to a send from the rank
        computing step 0 to the rank computing step 1.

        However, you can only receive what somebody sent and in order to make sure that is true for the non MPI
        versions, we reset after each iteration so you cannot use this function to communicate backwards from the last
        step to the first one for instance.

        This function is called both at the end of instantiating the controller, as well as after each iteration.

        Args:
            controller (pySDC.Controller): The controller

        Returns:
            None
        '''
        pass

    def post_iteration_processing(self, controller, S):
        '''
        Do whatever you want to after each iteration here.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        '''
        pass

    def post_step_processing(self, controller, S):
        '''
        Do whatever you want to after each step here.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        '''
        pass

    def prepare_next_block(self, controller, S, size, time, Tend):
        '''
        Prepare stuff like spreading step sizes or whatever.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step
            size (int): The number of ranks
            time (float): The current time
            Tend (float): The final time

        Returns:
            None
        '''
        pass

    def prepare_next_block_nonMPI(self, controller, MS, active_slots, time, Tend):
        '''
        This is an extension to the function `prepare_next_block`, which is only called in the non MPI controller and
        is needed because there is no chance to communicate backwards otherwise. While you should not do this in the
        first place, the first step in the new block comes after the last step in the last block, such that it is still
        in fact forwards communication, even though it looks backwards.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step
            active_slots (list): Index list of active steps
            time (float): The current time
            Tend (float): The final time

        Returns:
            None
        '''
        pass

    def convergence_control(self, controller, S):
        '''
        Call all the functions related to convergence control.
        This is called in `it_check` in the controller after every iteration just after `post_iteration_processing`.
        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        '''

        self.get_new_step_size(controller, S)
        self.determine_restart(controller, S)
        self.check_iteration_status(controller, S)

        return None

    def post_spread_processing(self, controller, S):
        '''
        This function is called at the end of the `SPREAD` stage in the controller

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step
        '''
        pass
