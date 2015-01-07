import copy as cp
import logging

from pySDC import Stats as statclass


class level():
    """
    Level class containing all management functionality for a single level

    A level contains all data structures, types and objects to perform sweeps on this particular level. It does not
    know about other levels.

    Attributes:
        __sweep: a private instance of a sweeper class (accessed via property)
        __prob: a private instance of a problem class (accessed via property)
        params: parameter object containing the custom parameters passed by the user
        status: status object
        uend: dof values at the right end point of the interval
        u: dof values at the nodes
        f: RHS values at the nodes
        tau: FAS correction, allocated via step class if necessary
        id: custom string naming this level
        logger: a logging object for level-dependent output
        __step: link to the step where this level is part of (set from the outside by the step)
        __hooks: a private instance of a hooks class
        __slots__: list of attributes to avoid accidential creation of new class attributes
    """

    class cstatus():
        """
        Helper class for status objects

        Attributes:
            residual: current residual
            unlocked: indicates if the data on this level can be used
            updated: indicates if the data on this level is new
        """

        def __init__(self):
            """
            Initialization routine

            """

            self.residual = None #FIXME: isn't that obsolete?
            self.unlocked = False
            self.updated = False


    __slots__ = ('__prob','__sweep','uend','u','f','tau','status','params','id','__step','id','__tag','__hooks')


    def __init__(self, problem_class, problem_params, dtype_u, dtype_f, collocation_class, num_nodes, sweeper_class,
                 level_params, hook_class, id):
        """
        Initialization routine

        Args:
            problem_class: problem class
            problem_params: parameters for the problem to be initialized
            dtype_u: data type of the dofs
            dtype_f: data type of the RHS
            collocation_class: collocation class for the sweeper
            num_nodes: the only parameter for collocation class
            sweeper_class: sweeper class
            level_params: parameters given by the user, will be added as attributes
            hook_class: class to add hooks (e.g. for output and diag)
            id: custom string naming this level
        """

        # short helper class to add params as attributes
        class pars():
            def __init__(self,params):
                for k,v in params.items():
                    setattr(self,k,v)

        # instantiate collocation, sweeper, problem and hooks
        coll = collocation_class(num_nodes,0,1)
        self.__sweep = sweeper_class(coll)
        self.__prob = problem_class(problem_params,dtype_u,dtype_f)
        self.__hooks = hook_class()

        # set level parameters and status
        self.params = pars(level_params)
        self.status = level.cstatus()

        # empty data the nodes, the right end point and tau
        self.uend = None
        self.u = [None] * (self.sweep.coll.num_nodes+1)
        self.f = [None] * (self.sweep.coll.num_nodes+1)
        self.tau = None

        # set name
        self.id = id

        # dummy step variable, will be defined by registration at step
        self.__step = None

        # pass this level to the sweeper for easy access
        self.sweep._sweeper__set_level(self)
        self.hooks._hooks__set_level(self)

        self.__tag = None


    def reset_level(self):
        """
        Routine to clean-up the level for the next time step
        """

        # reset status
        self.status = level.cstatus()

        # all data back to None
        self.uend = None
        self.u = [None] * (self.sweep.coll.num_nodes+1)
        self.f = [None] * (self.sweep.coll.num_nodes+1)


    def __add_tau(self):
        """
        Routine to add memory for the FAS correction

        This will be called by the step if this level is not the finest one.
        """

        if self.tau is None:
            self.tau = [None] * self.sweep.coll.num_nodes
        else:
            raise WTF #FIXME


    def __set_step(self,S):
        """
        Defines the step this level belongs to (no explicit setter)
        """
        self.__step = S

    @property
    def sweep(self):
        """
        Getter for the sweeper
        """
        return self.__sweep

    @property
    def hooks(self):
        """
        Getter for the hooks
        """
        return self.__hooks

    @property
    def prob(self):
        """
        Getter for the problem
        """
        return self.__prob

    @property
    def time(self):
        """
        Meta-getter for the current time (only passing the step's time)
        """
        return self.__step.status.time

    @property
    def dt(self):
        """
        Meta-getter for the step size (only passing the step's step size)
        """
        return self.__step.status.dt

    @property
    def iter(self):
        """
        Meta-getter for the iteration (only passing the step's iteration)
        """
        return self.__step.status.iter

    @property
    def dt(self):
        """
        Meta-getter for the step size (only passing the step's step size)
        """
        return self.__step.status.dt

    @property
    def tag(self):
        """
        Getter for tag
        Returns:
            tag
        """
        return self.__tag

    @tag.setter
    def tag(self,t):
        """
        Setter for tag
        Args:
            s: new tag
        """
        self.__tag = t