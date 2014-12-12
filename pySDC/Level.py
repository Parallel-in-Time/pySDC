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
        stats: level status object
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


    __slots__ = ('__prob','__sweep','uend','u','f','tau','status','params','id','logger','__step','id','stats')


    def __init__(self, problem_class, problem_params, dtype_u, dtype_f, collocation_class, num_nodes, sweeper_class,
                 level_params, id):
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
            id: custom string naming this level
        """

        # short helper class to add params as attributes
        class pars():
            def __init__(self,params):
                for k,v in params.items():
                    setattr(self,k,v)

        # instantiate collocation, sweeper and problem
        coll = collocation_class(num_nodes,0,1)
        self.__sweep = sweeper_class(coll)
        self.__prob = problem_class(problem_params,dtype_u,dtype_f)

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

        # set logger, empty time
        self.logger = self.__create_logger()
        self.__change_logger('')

        # dummy step variable, will be defined by registration at step
        self.__step = None

        # set stats
        self.stats = statclass.level_stats()

        # pass this level to the sweeper for easy access
        self.sweep._sweeper__set_level(self)


    def __create_logger(self):
        """
        Routine to create a logging object

        Returns:
            logger: custom logger for this level
        """

        # some logger magic
        formatter = logging.Formatter('%(name)s at time %(simtime)8.4e -- %(levelname)s: %(message)s (logged at %(asctime)s)',
                                      datefmt='%d.%m.%Y %H:%M:%S')
        logger = logging.getLogger(str(self.id))
        # this is where the logging level is set
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        return logger


    def __change_logger(self,var):
        """
        Add custom text to logger, e.g. time

        The var input is put into the simtime attribute via the ContextFilter class. In the Formatter, the variable
        simtime was defined/prepared.

        Args:
            var: this will be part of the output
        """

        class ContextFilter(logging.Filter):
            """
            This is a filter which injects contextual information into the log.
            """
            def filter(self, record):
                record.simtime = var
                return True

        self.logger.addFilter(ContextFilter())

    def reset_level(self):
        """
        Routine to clean-up the level for the next time step
        """

        # reset status and stats
        self.status = level.cstatus()
        self.stats = statclass.level_stats()

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
        return self.__step.time

    @property
    def dt(self):
        """
        Meta-getter for the step size (only passing the step's step size)
        """
        return self.__step.dt

    @property
    def iter(self):
        """
        Meta-getter for the iteration (only passing the step's iteration)
        """
        return self.__step.iter