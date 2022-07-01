import numpy as np

from pySDC.core.ConvergenceController import ConvergenceController


class HotRod(ConvergenceController):
    """
    See for the reference:
    Lightweight and Accurate Silent Data Corruption Detection in Ordinary Differential Equation Solvers,
    Guhur et al. 2016, Springer. DOI: 10.1007/978-3-319-43659-3_47
    """

    def __init__(self, controller, description, order):
        convergence_control_params = description.get('convergence_control_params', {})
        convergence_control_params['HotRod_tol'] = convergence_control_params.get('HotRod_tol', np.inf)
        super(HotRod, self).__init__(controller, description, order)

    def check_parameters(self, controller, description):
        '''
        Check whether parameters are compatible with whatever assumptions went into the step size functions etc.

        Returns:
            Whether the parameters are compatible
        '''
        if self.params.HotRod_tol == np.inf:
            controller.logger.warning('Hot Rod needs a detection threshold, which is now set to infinity, such that a restart\
 is never triggered!')

        if description['step_params'].get('restol', -1.) >= 0:
            return False, 'Hot Rod needs constant order in time and hence restol in the step parameters has to be \
smaller than 0!'

        if controller.params.mssdc_jac:
            return False, 'Hot Rod needs the same order on all steps, please activate Gauss-Seidel multistep mode!'

        if not controller.params.use_embedded_estimate:
            return False, 'Hot Rod needs embedded estimate, please turn it on!'

        if not controller.params.use_extrapolation_estimate:
            return False, 'Hot Rod needs extrapolation estimate, please turn it on!'

        return True, ''

    def determine_restart(self, controller, S):
        '''
        Determine for each step separately if it wants to be restarted for whatever reason. All steps after this one
        will be recomputed also.
        '''
        for L in S.levels:
            if None not in [L.status.error_extrapolation_estimate, L.status.error_embedded_estimate]:
                diff = L.status.error_extrapolation_estimate - L.status.error_embedded_estimate
                if diff > self.params.HotRod_tol:
                    S.status.restart = True
        super(HotRod, self).determine_restart(controller, S)

    def post_iteration_processing(self, controller, S):
        '''
        Throw away the final sweep to match the error estimates.
        '''
        if S.status.iter == S.params.maxiter:
            for L in S.levels:
                L.u[:] = L.uold[:]
