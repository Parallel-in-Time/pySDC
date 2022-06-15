from pySDC.helpers.pysdc_helper import FrozenClass


# short helper class to add params as attributes
class _Pars(FrozenClass):
    def __init__(self, params):
        self.order = 0

        for k, v in params.items():
            setattr(self, k, v)

        self._freeze()


class ConvergenceController(object):
    """
    Base abstract class for convergence controller, which is plugged into the controller to determine the iteration
    count and time step size.
    """

    def __init__(self, controller, description):
        self.params = _Pars(description.get('convergence_control_params', {}))
        params_ok, msg = self.check_parameters(controller, description)
        assert params_ok, msg
        self.reset_global_variables(controller)

    def check_parameters(self, controller, description):
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
        Determine for each step separately if it wants to be restarted for whatever reason. All steps after this one
        will be recomputed also.
        '''
        S.status.restart = S.status.restart or self.restart
        self.restart = self.restart or S.status.restart

    def reset_global_variables(self, controller):
        '''
        Global variables refer to variables used accross multiple steps that are stored in the convergence controller
        classes to immitate communication in non mpi versions. These have to be resetted in order to replicate
        avalability of variables in mpi versions.
        '''
        self.restart = False

    def post_iteration_processing(self, controller, S):
        '''
        Do whatever after the iteration here.
        '''
        pass
