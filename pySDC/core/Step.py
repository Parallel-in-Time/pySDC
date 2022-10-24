import logging

from pySDC.core import Level as levclass
from pySDC.core.BaseTransfer import base_transfer
from pySDC.core.Errors import ParameterError
from pySDC.helpers.pysdc_helper import FrozenClass


# short helper class to add params as attributes
class _Pars(FrozenClass):
    def __init__(self, params):
        self.maxiter = None
        for k, v in params.items():
            setattr(self, k, v)
        # freeze class, no further attributes allowed from this point
        self._freeze()


# short helper class to bundle all status variables
class _Status(FrozenClass):
    def __init__(self):
        self.iter = None
        self.stage = None
        self.slot = None
        self.first = None
        self.last = None
        self.pred_cnt = None
        self.done = None
        self.force_done = None
        self.prev_done = None
        self.time_size = None
        self.diff_old_loc = None
        self.diff_first_loc = None
        # freeze class, no further attributes allowed from this point
        self._freeze()


class step(FrozenClass):
    """
    Step class, referencing most of the structure needed for the time-stepping

    This class bundles multiple levels and the corresponding transfer operators and is used by the controller
    (e.g. SDC and MLSDC). Status variables like the current time are hidden via properties and setters methods.

    Attributes:
        params (__Pars): parameters given by the user
        status (__Status): status class for the step
        logger: custom logger for step-related logging
        levels (list): list of levels
    """

    def __init__(self, description):
        """
        Initialization routine

        Args:
            description (dict): parameters given by the user, will be added as attributes
        """

        # set params and status
        self.params = _Pars(description.get('step_params', {}))
        self.status = _Status()

        # set up logger
        self.logger = logging.getLogger('step')

        # empty attributes
        self.__transfer_dict = {}
        self.base_transfer = None
        self.levels = []
        self.__prev = None
        self.__next = None

        # freeze class, no further attributes allowed from this point
        self._freeze()

        # create hierarchy of levels
        self.__generate_hierarchy(description)

    def __generate_hierarchy(self, descr):
        """
        Routine to generate the level hierarchy for a single step

        This makes the explicit generation of levels in the frontend obsolete and hides a few dirty hacks here and
        there.

        Args:
            descr (dict): dictionary containing the description of the levels as list per key
        """

        if 'dtype_u' in descr:
            raise ParameterError(
                'Deprecated parameter dtype_u, please remove from description dictionary and specify '
                'directly in the problem class'
            )
        if 'dtype_f' in descr:
            raise ParameterError(
                'Deprecated parameter dtype_f, please remove from description dictionary and specify '
                'directly in the problem class'
            )

        # assert the existence of all the keys we need to set up at least on level
        essential_keys = ['problem_class', 'sweeper_class', 'sweeper_params', 'level_params']
        for key in essential_keys:
            if key not in descr:
                msg = 'need %s to instantiate step, only got %s' % (key, str(descr.keys()))
                self.logger.error(msg)
                raise ParameterError(msg)

        descr['problem_params'] = descr.get('problem_params', {})
        # check if base_transfer class is specified
        descr['base_transfer_class'] = descr.get('base_transfer_class', base_transfer)
        # check if base_transfer parameters are needed
        descr['base_transfer_params'] = descr.get('base_transfer_params', {})
        # check if space_transfer class is specified
        descr['space_transfer_class'] = descr.get('space_transfer_class', {})
        # check if space_transfer parameters are needed
        descr['space_transfer_params'] = descr.get('space_transfer_params', {})

        # convert problem-dependent parameters consisting of dictionary of lists to a list of dictionaries with only a
        # single entry per key, one dict per level
        pparams_list = self.__dict_to_list(descr['problem_params'])
        lparams_list = self.__dict_to_list(descr['level_params'])
        swparams_list = self.__dict_to_list(descr['sweeper_params'])
        # put this newly generated list into the description dictionary (copy to avoid changing the original one)
        descr_new = descr.copy()
        descr_new['problem_params'] = pparams_list
        descr_new['level_params'] = lparams_list
        descr_new['sweeper_params'] = swparams_list
        # generate list of dictionaries out of the description
        descr_list = self.__dict_to_list(descr_new)

        # sanity check: is there a base_transfer class? is there one even if only a single level is specified?
        if len(descr_list) > 1 and not descr_new['space_transfer_class']:
            msg = 'need %s to instantiate step, only got %s' % ('space_transfer_class', str(descr_new.keys()))
            self.logger.error(msg)
            raise ParameterError(msg)

        if len(descr_list) == 1 and (
            descr_new['space_transfer_class'] or descr_new['base_transfer_class'] is not base_transfer
        ):
            self.logger.warning('you have specified transfer classes, but only a single level')

        # generate levels, register and connect if needed
        for l in range(len(descr_list)):

            L = levclass.level(
                problem_class=descr_list[l]['problem_class'],
                problem_params=descr_list[l]['problem_params'],
                sweeper_class=descr_list[l]['sweeper_class'],
                sweeper_params=descr_list[l]['sweeper_params'],
                level_params=descr_list[l]['level_params'],
                level_index=l,
            )

            self.levels.append(L)

            if l > 0:
                self.connect_levels(
                    base_transfer_class=descr_new['base_transfer_class'],
                    base_transfer_params=descr_list[l]['base_transfer_params'],
                    space_transfer_class=descr_list[l]['space_transfer_class'],
                    space_transfer_params=descr_list[l]['space_transfer_params'],
                    fine_level=self.levels[l - 1],
                    coarse_level=self.levels[l],
                )

    @staticmethod
    def __dict_to_list(in_dict):
        """
        Straightforward helper function to convert dictionary of list to list of dictionaries

        Args:
            in_dict (dict): dictionary of lists
        Returns:
            list of dictionaries
        """

        max_val = 1
        for _, v in in_dict.items():
            if type(v) is list:
                max_val = max(max_val, len(v))
            else:
                pass

        ld = [{} for _ in range(max_val)]
        for d in range(len(ld)):
            for k, v in in_dict.items():
                if type(v) is not list:
                    ld[d][k] = v
                else:
                    ld[d][k] = v[min(d, len(v) - 1)]
        return ld

    def connect_levels(
        self,
        base_transfer_class,
        base_transfer_params,
        space_transfer_class,
        space_transfer_params,
        fine_level,
        coarse_level,
    ):
        """
        Routine to couple levels with base_transfer operators

        Args:
            base_transfer_class: the class which can do transfer between the two space-time levels
            base_transfer_params (dict): parameters for the space_transfer class
            space_transfer_class: the user-defined class which can do spatial transfer
            space_transfer_params (dict): parameters for the base_transfer class
            fine_level (pySDC.Level.level): the fine level
            coarse_level (pySDC.Level.level): the coarse level
        """

        # create new instance of the specific base_transfer class
        self.base_transfer = base_transfer_class(
            fine_level, coarse_level, base_transfer_params, space_transfer_class, space_transfer_params
        )
        # use base_transfer dictionary twice to set restrict and prolong operator
        self.__transfer_dict[(fine_level, coarse_level)] = self.base_transfer.restrict

        if self.base_transfer.params.finter:
            self.__transfer_dict[(coarse_level, fine_level)] = self.base_transfer.prolong_f
        else:
            self.__transfer_dict[(coarse_level, fine_level)] = self.base_transfer.prolong

    def transfer(self, source, target):
        """
        Wrapper routine to ease the call of the transfer functions

        This function can be called in the multilevel stepper (e.g. MLSDC), passing a source and a target level.
        Using the transfer dictionary, the calling stepper does not need to specify whether to use restrict of
        prolong.

        Args:
            source (pySDC.Level.level): source level
            target (pySDC.Level.level): target level
        """
        self.__transfer_dict[(source, target)]()

    def reset_step(self):
        """
        Routine so clean-up step structure and the corresp. levels for further uses
        """
        # reset all levels
        for l in self.levels:
            l.reset_level()

    def init_step(self, u0):
        """
        Initialization routine for a new step.

        This routine uses initial values u0 to set up the u[0] values at the finest level

        Args:
            u0 (dtype_u): initial values
        """

        assert len(self.levels) >= 1
        assert len(self.levels[0].u) >= 1

        # pass u0 to u[0] on the finest level 0
        P = self.levels[0].prob
        self.levels[0].u[0] = P.dtype_u(u0)

    @property
    def prev(self):
        """
        Getter for previous step

        Returns:
            prev
        """
        return self.__prev

    @prev.setter
    def prev(self, p):
        """
        Setter for previous step

        Args:
            p: new previous step
        """
        self.__prev = p

    @property
    def next(self):
        """
        Getter for next step

        Returns:
            prev
        """
        return self.__next

    @next.setter
    def next(self, p):
        """
        Setter for next step

        Args:
            p: new next step
        """
        self.__next = p

    @property
    def dt(self):
        """
        Getter for current time-step size

        Returns:
            float: dt of level[0]
        """
        return self.levels[0].dt

    @property
    def time(self):
        """
        Getter for current time

        Returns:
            float: time of level[0]
        """
        return self.levels[0].time
