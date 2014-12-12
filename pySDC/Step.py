from pySDC import Level as levclass
from pySDC import Stats as statclass

class step():
    """
    Step class, referencing most of the structure needed for the time-stepping

    This class bundles multiple levels and the corresponding transfer operators and is used by the methods
    (e.g. SDC and MLSDC). Status variables like the current time are hidden via properties and setters methods.

    Attributes:
        __t: current time (property time)
        __dt: current step size (property dt)
        __k: current iteration (property iter)
        __transfer_dict: data structure to couple levels and transfer operators
        levels: list of levels
        stats: step statistics
        params: parameters given by the user
        __slots__: list of attributes to avoid accidential creation of new class attributes
    """

    __slots__ = ('params','stats','__t','__dt','__k','levels','__transfer_dict')

    def __init__(self, params):
        """
        Initialization routine

        Args:
            params: parameters given by the user, will be added as attributes
        """

        # short helper class to add params as attributes
        class pars():
            def __init__(self,params):
                for k,v in params.items():
                    setattr(self,k,v)
                pass

        # set params and stats
        self.params = pars(params)
        self.stats = statclass.step_stats()

        # empty attributes
        self.__t = None
        self.__dt = None
        self.__k = None
        self.__transfer_dict = {}
        self.levels = []


    def generate_hierarchy(self,descr):
        """
        Routine to generate the level hierarchy for a single step

        This makes the explicit generation of levels in the frontend obsolete and hides a few dirty hacks here and
        there.

        Args:
            descr: dictionary containing the description of the levels as list per key
        """
        # assert the existence of all the keys we need to set up at least on level
        assert 'problem_class' in descr
        assert 'problem_params' in descr
        assert 'dtype_u' in descr
        assert 'dtype_f' in descr
        assert 'collocation_class' in descr
        assert 'num_nodes' in descr
        assert 'sweeper_class' in descr
        assert 'level_params' in descr

        # convert problem-dependent parameters consisting of dictionary of lists to a list of dictionaries with only a
        # single entry per key, one dict per level
        pparams_list = self.__dict_to_list(descr['problem_params'])
        # put this newly generated list into the description dictionary
        descr['problem_params'] = pparams_list
        # generate list of dictionaries out of the description
        descr_list = self.__dict_to_list(descr)

        # sanity check: is there a transfer class? is there one even if only a single level is specified?
        if len(descr_list) > 1:
            assert 'transfer_class' in descr
        elif 'transfer_class' in descr:
            print('WARNING: you have specified transfer classes, but only a single level...')

        # generate levels, register and connect if needed
        for l in range(len(pparams_list)):

            L = levclass.level(problem_class      =   descr_list[l]['problem_class'],
                               problem_params     =   descr_list[l]['problem_params'],
                               dtype_u            =   descr_list[l]['dtype_u'],
                               dtype_f            =   descr_list[l]['dtype_f'],
                               collocation_class  =   descr_list[l]['collocation_class'],
                               num_nodes          =   descr_list[l]['num_nodes'],
                               sweeper_class      =   descr_list[l]['sweeper_class'],
                               level_params       =   descr_list[l]['level_params'],
                               id                 =   'L'+str(l))

            self.register_level(L)

            if l > 0:
                self.connect_levels(transfer_class = descr_list[l]['transfer_class'],
                                    fine_level     = self.levels[l-1],
                                    coarse_level   = self.levels[l])


    @staticmethod
    def __dict_to_list(dict):
        """
        Straightforward helper function to convert dictionary of list to list of dictionaries

        Args:
            dict: dictionary of lists
        Returns:
            list of dictionaries
        """

        max_val = 1
        for k,v in dict.items():
            if type(v) is list:
                if max_val > 1 and len(v) is not max_val:
                    sys.exit('All lists in cparams need to be of length 1 or %i.. key %s has this list: %s' %(max_val,k,v))
                max_val = len(v)

        ld = [{} for l in range(max_val)]
        for d in range(len(ld)):
            for k,v in dict.items():
                if type(v) is not list:
                    ld[d][k] = v
                else:
                    ld[d][k] = v[d]
        return ld


    def register_level(self,L):
        """
        Routine to register levels

        This routine will append levels to the level list of the step instance and link the step to the newly
        registered level (Level 0 will be considered as the finest level). It will also allocate the tau correction,
        if this level is not the finest one.

        Args:
            L: level to be registered
        """

        assert isinstance(L,levclass.level)
        # add level to level list
        self.levels.append(L)
        # pass this step to the registered level
        self.levels[-1]._level__set_step(self)
        # if this is not the finest level, allocate tau correction
        if len(self.levels) > 1:
            L._level__add_tau()


    def connect_levels(self, transfer_class, fine_level, coarse_level):
        """
        Routine to couple levels with transfer operators

        Args:
            transfer_class: the class which can transfer between he two levels
            fine_level: the fine level
            coarse_level: the coarse level
        """

        # create new instance of the specific transfer class
        T = transfer_class(fine_level,coarse_level)
        # use transfer dictionary twice to set restrict and prologn operator
        self.__transfer_dict[tuple([fine_level,coarse_level])] = T.restrict
        self.__transfer_dict[tuple([coarse_level,fine_level])] = T.prolong


    def transfer(self,source,target):
        """
        Wrapper routine to ease the call of the transfer functions

        This function can be called in the multilevel stepper (e.g. MLSDC), passing a source and a target level.
        Using the transfer dictionary, the calling stepper does not need to specify whether to use restrict of prolong.

        Args:
            source: source level
            target: target level
        """

        self.__transfer_dict[tuple([source,target])]()


    def reset_step(self):
        """
        Routine so clean-up step structure and the corresp. levels for further uses
        """

        # reset step statistics
        self.stats = statclass.step_stats()
        # reset all levels
        for l in self.levels:
            l.reset_level()


    def init_step(self,u0):
        """
        Initialization routine for a new step.

        This routine uses initial values u0 to set up the u[0] values at the finest level and the level stats are
        linked with the step stats for easy access.

        Args:
            u0: initial values
        """

        assert len(self.levels) >=1
        assert len(self.levels[0].u) >=1

        # link level stats to step stats and reset level status
        for L in self.levels:
            self.stats.register_level_stats(L.stats)

        # pass u0 to u[0] on the finest level 0
        P = self.levels[0].prob
        self.levels[0].u[0] = P.dtype_u(u0)


    @property
    def time(self):
        """
        Getter for __t/time
        Returns:
            time
        """
        return self.__t


    @time.setter
    def time(self,t):
        """
        Setter for __t/time
        Args:
            t: time to set
        """
        self.__t = t
        # pass new time to level loggers
        for l in self.levels:
            l._level__change_logger(self.__t)


    @property
    def dt(self):
        """
        Getter for __dt/dt
        Returns:
            dt
        """
        return self.__dt


    @dt.setter
    def dt(self,dt):
        """
        Setter for __dt/d
        Args:
            dt: step size to set
        """
        self.__dt = dt


    @property
    def iter(self):
        """
        Getter for __k/iter
        Returns:
            iter
        """
        return self.__k


    @iter.setter
    def iter(self,k):
        """
        Setter for __k/iter
        Args:
            k: iteration to set
        """
        self.__k = k