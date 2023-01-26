from pySDC.helpers.pysdc_helper import FrozenClass


# short helper class to add params as attributes
class _Pars(FrozenClass):
    def __init__(self, params):
        self.dt = None
        self.dt_initial = None
        self.restol = -1.0
        self.nsweeps = 1
        self.residual_type = 'full_abs'
        for k, v in params.items():
            setattr(self, k, v)
        # freeze class, no further attributes allowed from this point
        self._freeze()

        self.dt_initial = self.dt * 1.0


# short helper class to bundle all status variables
class _Status(FrozenClass):
    """
    This class carries the status of the level. All variables that the core SDC / PFASST functionality depend on are
    initialized here, while the convergence controllers are allowed to add more variables in a controlled fashion
    later on using the `add_variable` function.
    """

    def __init__(self):
        self.residual = None
        self.unlocked = False
        self.updated = False
        self.time = None
        self.dt_new = None
        self.sweep = None
        # freeze class, no further attributes allowed from this point
        self._freeze()


class level(FrozenClass):
    """
    Level class containing all management functionality for a single level

    A level contains all data structures, types and objects to perform sweeps on this particular level. It does not
    know about other levels.

    Attributes:
        params (__Pars): parameter object containing the custom parameters passed by the user
        status (__Status): status object
        level_index (int): custom string naming this level
        uend: dof values at the right end point of the interval
        u (list of dtype_u): dof values at the nodes
        uold (list of dtype_u): copy of dof values for saving data during restriction)
        f (list of dtype_f): RHS values at the nodes
        fold (list of dtype_f): copy of RHS values for saving data during restriction
        tau (list of dtype_u): FAS correction, allocated via step class if necessary
    """

    def __init__(self, problem_class, problem_params, sweeper_class, sweeper_params, level_params, level_index):
        """
        Initialization routine

        Args:
            problem_class: problem class
            problem_params (dict): parameters for the problem to be initialized
            sweeper_class: sweeper class
            sweeper_params (dict): parameters for the sweeper (contains collocation)
            level_params (dict): parameters given by the user, will be added as attributes
            level_index (int): custom name for this level
        """

        # instantiate sweeper, problem and hooks
        self.__sweep = sweeper_class(sweeper_params)
        self.__prob = problem_class(problem_params)

        # set level parameters and status
        self.params = _Pars(level_params)
        self.status = _Status()

        # set name
        self.level_index = level_index

        # empty data at the nodes, the right end point and tau
        self.uend = None
        self.u = [None] * (self.sweep.coll.num_nodes + 1)
        self.uold = [None] * (self.sweep.coll.num_nodes + 1)
        self.f = [None] * (self.sweep.coll.num_nodes + 1)
        self.fold = [None] * (self.sweep.coll.num_nodes + 1)

        self.tau = [None] * self.sweep.coll.num_nodes

        # pass this level to the sweeper for easy access
        self.sweep.level = self

        self.__tag = None

        # freeze class, no further attributes allowed from this point
        self._freeze()

    def reset_level(self, reset_status=True):
        """
        Routine to clean-up the level for the next time step

        Args:
            reset_status (bool): Reset the status or only the solution

        Returns:
            None
        """

        # reset status
        if reset_status:
            self.status = _Status()

        # all data back to None
        self.uend = None
        self.u = [None] * (self.sweep.coll.num_nodes + 1)
        self.uold = [None] * (self.sweep.coll.num_nodes + 1)
        self.f = [None] * (self.sweep.coll.num_nodes + 1)
        self.fold = [None] * (self.sweep.coll.num_nodes + 1)
        self.tau = [None] * self.sweep.coll.num_nodes

    @property
    def sweep(self):
        """
        Getter for the sweeper

        Returns:
            pySDC.Sweeper.sweeper: the sweeper associated to this level
        """
        return self.__sweep

    @property
    def prob(self):
        """
        Getter for the problem

        Returns:
            pySDC.Problem.ptype: the problem associated to this level
        """
        return self.__prob

    @property
    def time(self):
        """
        Meta-getter for the current time

        Returns:
            float: referencing status time for convenience
        """
        return self.status.time

    @property
    def dt(self):
        """
        Meta-getter for the time-step size

        Returns:
            float: referencing dt from parameters for convenience
        """
        return self.params.dt

    @property
    def tag(self):
        """
        Getter for tag

        Returns:
            tag for sending/receiving
        """
        return self.__tag

    @tag.setter
    def tag(self, t):
        """
        Setter for tag

        Args:
            t: new tag for sending/receiving
        """
        self.__tag = t
