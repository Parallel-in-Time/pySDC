import logging
from pySDC.helpers.pysdc_helper import FrozenClass


# short helper class to add params as attributes
class Pars(FrozenClass):
    def __init__(self, params):
        self.control_order = 0  # integer that determines the order in which the convergence controllers are called
        self.useMPI = None  # depends on the controller

        for k, v in params.items():
            setattr(self, k, v)

        self._freeze()


# short helper class to store status variables
class Status(FrozenClass):
    """
    Initialize status variables with None, since at the time of instantiation of the convergence controllers, not all
    relevant information about the controller are known.
    """

    def __init__(self, status_variabes):
        [setattr(self, key, None) for key in status_variabes]

        self._freeze()


class ConvergenceController(object):
    """
    Base abstract class for convergence controller, which is plugged into the controller to determine the iteration
    count and time step size.
    """

    def __init__(self, controller, params, description, **kwargs):
        """
        Initialization routine

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller
        """
        self.controller = controller
        self.params = Pars(self.setup(controller, params, description))
        params_ok, msg = self.check_parameters(controller, params, description)
        assert params_ok, f'{type(self).__name__} -- {msg}'
        self.dependencies(controller, description)
        self.logger = logging.getLogger(f"{type(self).__name__}")

        if self.params.useMPI:
            self.prepare_MPI_datatypes()

    def prepare_MPI_logical_operations(self):
        """
        Prepare MPI logical operations so we don't need to import mpi4py all the time
        """
        from mpi4py import MPI

        self.MPI_LAND = MPI.LAND
        self.MPI_LOR = MPI.LOR

    def prepare_MPI_datatypes(self):
        """
        Prepare MPI datatypes so we don't need to import mpi4py all the time
        """
        from mpi4py import MPI

        self.MPI_INT = MPI.INT
        self.MPI_DOUBLE = MPI.DOUBLE
        self.MPI_BOOL = MPI.BOOL

    def log(self, msg, S, level=15, **kwargs):
        """
        Shortcut that has a default level for the logger. 15 is above debug but below info.

        Args:
            msg (str): Message you want to log
            S (pySDC.step): The current step
            level (int): the level passed to the logger

        Returns:
            None
        """
        self.logger.log(level, f'Process {S.status.slot:2d} on time {S.time:.6f} - {msg}')
        return None

    def debug(self, msg, S, **kwargs):
        """
        Shortcut to pass messages at debug level to the logger.

        Args:
            msg (str): Message you want to log
            S (pySDC.step): The current step

        Returns:
            None
        """
        self.log(msg=msg, S=S, level=10, **kwargs)
        return None

    def setup(self, controller, params, description, **kwargs):
        """
        Setup various variables that only need to be set once in the beginning.
        If the convergence controller is added automatically, you can give it params by adding it manually.
        It will be instantiated only once with the manually supplied parameters overriding automatically added
        parameters.

        This function scans the convergence controllers supplied to the description object for instances of itself.
        This corresponds to the convergence controller being added manually by the user. If something is found, this
        function will then return a composite dictionary from the `params` passed to this function as well as the
        `params` passed manually, with priority to manually added parameters. If you added the convergence controller
        manually, that is of course the same and nothing happens. If, on the other hand, the convergence controller was
        added automatically, the `params` passed here will come from whatever added it and you can now override
        parameters by adding the convergence controller manually.
        This relies on children classes to return a composite dictionary from their defaults and from the result of this
        function, so you should write
        ```
        return {**defaults, **super().setup(controller, params, description, **kwargs)}
        ```
        when overloading this method in a child class, with `defaults` a dictionary containing default parameters.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary after setup
        """
        # allow to change parameters by adding the convergence controller manually
        return {**params, **description.get('convergence_controllers', {}).get(type(self), {})}

    def dependencies(self, controller, description, **kwargs):
        """
        Load dependencies on other convergence controllers here.

        Args:
            controller (pySDC.Controller): The controller
            description (dict): The description object used to instantiate the controller

        Returns:
            None
        """
        pass

    def check_parameters(self, controller, params, description, **kwargs):
        """
        Check whether parameters are compatible with whatever assumptions went into the step size functions etc.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            bool: Whether the parameters are compatible
            str: The error message
        """
        return True, ""

    def check_iteration_status(self, controller, S, **kwargs):
        """
        Determine whether to keep iterating or not in this function.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """
        pass

    def get_new_step_size(self, controller, S, **kwargs):
        """
        This function allows to set a step size with arbitrary criteria.
        Make sure to give an order to the convergence controller by setting the `control_order` variable in the params.
        This variable is an integer and you can see what the current order is by using
        `controller.print_convergence_controllers()`.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """
        pass

    def determine_restart(self, controller, S, **kwargs):
        """
        Determine for each step separately if it wants to be restarted for whatever reason.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """
        pass

    def reset_status_variables(self, controller, **kwargs):
        """
        Reset status variables.
        This is called in the `restart_block` function.
        Args:
            controller (pySDC.Controller): The controller

        Returns:
            None
        """
        return None

    def setup_status_variables(self, controller, **kwargs):
        """
        Setup status variables.
        This is not done at the time of instantiation, since the controller is not fully instantiated at that time and
        hence not all information are available. Instead, this function is called after the controller has been fully
        instantiated.

        Args:
            controller (pySDC.Controller): The controller

        Returns:
            None
        """
        return None

    def reset_buffers_nonMPI(self, controller, **kwargs):
        """
        Buffers refer to variables used across multiple steps that are stored in the convergence controller classes to
        imitate communication in non MPI versions. These have to be reset in order to replicate availability of
        variables in MPI versions.

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
        """
        pass

    def pre_iteration_processing(self, controller, S, **kwargs):
        """
        Do whatever you want to before each iteration here.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """
        pass

    def post_iteration_processing(self, controller, S, **kwargs):
        """
        Do whatever you want to after each iteration here.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """
        pass

    def post_step_processing(self, controller, S, **kwargs):
        """
        Do whatever you want to after each step here.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """
        pass

    def prepare_next_block(self, controller, S, size, time, Tend, **kwargs):
        """
        Prepare stuff like spreading step sizes or whatever.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step
            size (int): The number of ranks
            time (float): The current time will be list in nonMPI controller implementation
            Tend (float): The final time

        Returns:
            None
        """
        pass

    def convergence_control(self, controller, S, **kwargs):
        """
        Call all the functions related to convergence control.
        This is called in `it_check` in the controller after every iteration just after `post_iteration_processing`.
        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """

        self.get_new_step_size(controller, S, **kwargs)
        self.determine_restart(controller, S, **kwargs)
        self.check_iteration_status(controller, S, **kwargs)

        return None

    def post_spread_processing(self, controller, S, **kwargs):
        """
        This function is called at the end of the `SPREAD` stage in the controller

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step
        """
        pass

    def send(self, comm, dest, data, blocking=False, **kwargs):
        """
        Send data to a different rank

        Args:
            comm (mpi4py.MPI.Intracomm): Communicator
            dest (int): The target rank
            data: Data to be sent
            blocking (bool): Whether the communication is blocking or not

        Returns:
            request handle of the communication
        """
        kwargs['tag'] = kwargs.get('tag', abs(self.params.control_order))

        # log what's happening for debug purposes
        self.logger.debug(f'Step {comm.rank} {"" if blocking else "i"}sends to step {dest} with tag {kwargs["tag"]}')

        if blocking:
            req = comm.send(data, dest=dest, **kwargs)
        else:
            req = comm.isend(data, dest=dest, **kwargs)

        return req

    def recv(self, comm, source, **kwargs):
        """
        Receive some data

        Args:
            comm (mpi4py.MPI.Intracomm): Communicator
            source (int): Where to look for receiving

        Returns:
            whatever has been received
        """
        kwargs['tag'] = kwargs.get('tag', abs(self.params.control_order))

        # log what's happening for debug purposes
        self.logger.debug(f'Step {comm.rank} receives from step {source} with tag {kwargs["tag"]}')

        data = comm.recv(source=source, **kwargs)

        return data

    def Send(self, comm, dest, buffer, blocking=False, **kwargs):
        """
        Send data to a different rank

        Args:
            comm (mpi4py.MPI.Intracomm): Communicator
            dest (int): The target rank
            buffer: Buffer for the data
            blocking (bool): Whether the communication is blocking or not

        Returns:
            request handle of the communication
        """
        kwargs['tag'] = kwargs.get('tag', abs(self.params.control_order))

        # log what's happening for debug purposes
        self.logger.debug(f'Step {comm.rank} {"" if blocking else "i"}Sends to step {dest} with tag {kwargs["tag"]}')

        if blocking:
            req = comm.Send(buffer, dest=dest, **kwargs)
        else:
            req = comm.Isend(buffer, dest=dest, **kwargs)

        return req

    def Recv(self, comm, source, buffer, **kwargs):
        """
        Receive some data

        Args:
            comm (mpi4py.MPI.Intracomm): Communicator
            source (int): Where to look for receiving

        Returns:
            whatever has been received
        """
        kwargs['tag'] = kwargs.get('tag', abs(self.params.control_order))

        # log what's happening for debug purposes
        self.logger.debug(f'Step {comm.rank} Receives from step {source} with tag {kwargs["tag"]}')

        data = comm.Recv(buffer, source=source, **kwargs)

        return data

    def add_status_variable_to_step(self, key, value=None):
        if type(self.controller).__name__ == 'controller_MPI':
            steps = [self.controller.S]
        else:
            steps = self.controller.MS

        steps[0].status.add_attr(key)

        if value is not None:
            self.set_step_status_variable(key, value)

    def set_step_status_variable(self, key, value):
        if type(self.controller).__name__ == 'controller_MPI':
            steps = [self.controller.S]
        else:
            steps = self.controller.MS

        for S in steps:
            S.status.__dict__[key] = value

    def add_status_variable_to_level(self, key, value=None):
        if type(self.controller).__name__ == 'controller_MPI':
            steps = [self.controller.S]
        else:
            steps = self.controller.MS

        steps[0].levels[0].status.add_attr(key)

        if value is not None:
            self.set_level_status_variable(key, value)

    def set_level_status_variable(self, key, value):
        if type(self.controller).__name__ == 'controller_MPI':
            steps = [self.controller.S]
        else:
            steps = self.controller.MS

        for S in steps:
            for L in S.levels:
                L.status.__dict__[key] = value
