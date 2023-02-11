import logging
from pySDC.helpers.pysdc_helper import FrozenClass


# short helper class to add params as attributes
class Pars(FrozenClass):
    def __init__(self, params):
        self.control_order = 0  # integer that determines the order in which the convergence controllers are called

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
        self.params = Pars(self.setup(controller, params, description))
        params_ok, msg = self.check_parameters(controller, params, description)
        assert params_ok, msg
        self.dependencies(controller, description)
        self.logger = logging.getLogger(f"{type(self).__name__}")

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
            time (float): The current time
            Tend (float): The final time

        Returns:
            None
        """
        pass

    def prepare_next_block_nonMPI(self, controller, MS, active_slots, time, Tend):
        """
        This is an extension to the function `prepare_next_block`, which is only called in the non MPI controller and
        is needed because there is no chance to communicate backwards otherwise. While you should not do this in the
        first place, the first step in the new block comes after the last step in the last block, such that it is still
        in fact forwards communication, even though it looks backwards.

        Args:
            controller (pySDC.Controller): The controller
            MS (list): All steps of the controller
            active_slots (list): Index list of active steps
            time (float): The current time
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
        # log what's happening for debug purposes
        self.logger.debug(f'Step {comm.rank} initiates send to step {dest}')

        if blocking:
            req = comm.send(data, dest=dest, **kwargs)
        else:
            req = comm.isend(data, dest=dest, **kwargs)

        # log what's happening for debug purposes
        self.logger.debug(f'Step {comm.rank} leaves send to step {dest}')

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
        # log what's happening for debug purposes
        self.logger.debug(f'Step {comm.rank} initiates receive from step {source}')

        data = comm.recv(source=source, **kwargs)

        # log what's happening for debug purposes
        self.logger.debug(f'Step {comm.rank} leaves receive from step {source}')

        return data

    def reset_variable(self, controller, name, MPI=False, place=None, where=None, init=None):
        """
        Utility function for resetting variables. This function will call the `add_variable` function with all the same
        arguments, but with `allow_overwrite = True`.

        Args:
            controller (pySDC.Controller): The controller
            name (str): The name of the variable
            MPI (bool): Whether to use MPI controller
            place (object): The object you want to reset the variable of
            where (list): List of strings containing a path to where you want to reset the variable
            init: Initial value of the variable

        Returns:
            None
        """
        self.add_variable(controller, name, MPI, place, where, init, allow_overwrite=True)

    def add_variable(self, controller, name, MPI=False, place=None, where=None, init=None, allow_overwrite=False):
        """
        Add a variable to a frozen class.

        This function goes through the path to the destination of the variable recursively and adds it to all instances
        that are possible in the path. For example, giving `where = ["MS", "levels", "status"]` will result in adding a
        variable to the status object of all levels of all steps of the controller.

        Part of the functionality of the frozen class is to separate initialization and setting of variables. By
        enforcing this, you can make sure not to overwrite already existing variables. Since this function is called
        outside of the `__init__` function of the status objects, this can otherwise lead to bugs that are hard to find.
        For this reason, you need to specifically set `allow_overwrite = True` if you want to forgo the check if the
        variable already exists. This can be useful when resetting variables between steps, but make sure to set it to
        `allow_overwrite = False` the first time you add a variable.

        Args:
            controller (pySDC.Controller): The controller
            name (str): The name of the variable
            MPI (bool): Whether to use MPI controller
            place (object): The object you want to add the variable to
            where (list): List of strings containing a path to where you want to add the variable
            init: Initial value of the variable
            allow_overwrite (bool): Allow overwriting the variables if they already exist or raise an exception

        Returns:
            None
        """
        where = ["S" if MPI else "MS", "levels", "status"] if where is None else where
        place = controller if place is None else place

        # check if we have arrived at the end of the path to the variable
        if len(where) == 0:
            variable_exitsts = name in place.__dict__.keys()
            # check if the variable already exists and raise an error in case we are about to introduce a bug
            if not allow_overwrite and variable_exitsts:
                raise ValueError(f"Key \"{name}\" already exists in {place}! Please rename the variable in {self}")
            # if we allow overwriting, but the variable does not exist already, we are violating the intended purpose
            # of this function, so we also raise an error if someone should be so mad as to attempt this
            elif allow_overwrite and not variable_exitsts:
                raise ValueError(f"Key \"{name}\" is supposed to be overwritten in {place}, but it does not exist!")

            # actually add or overwrite the variable
            place.__dict__[name] = init

        # follow the path to the final destination recursively
        else:
            # get all possible new places to continue the path
            new_places = place.__dict__[where[0]]

            # continue all possible paths
            if type(new_places) == list:
                # loop through all possibilities
                for new_place in new_places:
                    self.add_variable(
                        controller,
                        name,
                        MPI=MPI,
                        place=new_place,
                        where=where[1:],
                        init=init,
                        allow_overwrite=allow_overwrite,
                    )
            else:
                # go to the only possible possibility
                self.add_variable(
                    controller,
                    name,
                    MPI=MPI,
                    place=new_places,
                    where=where[1:],
                    init=init,
                    allow_overwrite=allow_overwrite,
                )
