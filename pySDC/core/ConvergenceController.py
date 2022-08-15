import logging
from pySDC.helpers.pysdc_helper import FrozenClass


# short helper class to add params as attributes
class _Pars(FrozenClass):
    def __init__(self, params):
        self.control_order = 0

        for k, v in params.items():
            setattr(self, k, v)

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
        Setup various variables that only need to be set once in the beginning
        '''
        return params

    def dependencies(self, controller, description):
        '''
        Add dependencies in the form of other convergence controllers here
        '''
        pass

    def check_parameters(self, controller, params, description):
        '''
        Check whether parameters are compatible with whatever assumptions went into the step size functions etc.

        Returns:
            Whether the parameters are compatible
        '''
        return True, ''

    def check_iteration_status(self, controller, S):
        '''
        Determine whether to keep iterating or not in this function.
        '''
        pass

    def get_new_step_size(self, controller, S):
        '''
        This function allows to set a step size with arbitrary criteria. Make sure to give an order if you give
        multiple criteria. The order is a scalar, the higher, the later it is called meaning the highest order has the
        final word.
        '''
        pass

    def determine_restart(self, controller, S):
        '''
        Determine for each step separately if it wants to be restarted for whatever reason.
        '''
        pass

    def reset_global_variables_nonMPI(self, controller):
        '''
        Global variables refer to variables used accross multiple steps that are stored in the convergence controller
        classes to immitate communication in non mpi versions. These have to be resetted in order to replicate
        avalability of variables in mpi versions.
        '''
        pass

    def post_iteration_processing(self, controller, S):
        '''
        Do whatever after the iteration here.
        '''
        pass

    def post_step_processing(self, controller, S):
        '''
        Do whatever after the step here.
        '''
        pass

    def prepare_next_block(self, controller, S, size, time, Tend):
        '''
        You should take care here that two things happen:
          -  Every step after a step which wants to be restarted also needs to know it wants to be restarted
          -  Every step should somehow receive a new step size
        Of course not every convergence controller needs to implement this, just one
        '''
        pass

    def prepare_next_block_nonMPI(self, controller, MS, active_slots, time, Tend):
        '''
        This is an extension to the function `prepare_next_block`, which is only called in the non MPI controller and
        is needed because there is no chance to communicate backwards otherwise. While you should not do this in the
        first place, the first step in the new block comes after the last step in the last block, such that is still in
        fact forwards communication, even though it looks backwards.
        '''
        pass
