from pySDC.core.Convergence_Controller import ConvergenceController


class Adaptivity(ConvergenceController):
    """
    Class to compute time step size adaptively based on embedded error estimate.
    Adaptivity requires you to know the order of the scheme, which you can also know for Jacobi, but it works
    differently.
    """

    def __init__(self, controller, description):
        super(Adaptivity, self).__init__(controller, description)
        self.params.order = -90

    def check_parameters(self, controller, description):
        '''
        Check whether parameters are compatible with whatever assumptions went into the step size functions etc.

        Returns:
            Whether the parameters are compatible
            The error message
        '''
        if description['step_params'].get('restol', -1.) >= 0:
            return False, 'Adaptivity needs constant order in time and hence restol in the step parameters has to be \
smaller than 0!'

        if controller.params.mssdc_jac:
            return False, 'Adaptivity needs the same order on all steps, please activate Gauss-Seidel multistep mode!'

        if 'e_tol' not in description.get('convergence_control_params', {}).keys():
            return False, 'Adaptivity needs a local tolerance! Please set some up in description[\'convergence_control\
_params\'][\'e_tol\']!'

        return True, ''

    def get_new_step_size(self, controller, S):
        '''
        This function allows to set a step size with arbitrary criteria. Make sure to give an order if you give
        multiple criteria. The order is a scalar, the higher, the later it is called meaning the highest order has the
        final word.
        '''
        # check if we performed the desired amount of sweeps
        if S.status.iter == S.params.maxiter:
            L = S.levels[0]

            # compute next step size
            order = S.status.iter  # embedded error estimate is same order as time marching
            L.status.dt_new = L.params.dt * 0.9 * (self.params.e_tol / L.status.error_embedded_estimate)**(1. / order)

    def determine_restart(self, controller, S):
        '''
        Determine for each step separately if it wants to be restarted for whatever reason. All steps after this one
        will be recomputed also.
        '''
        if S.status.iter == S.params.maxiter:
            if S.levels[0].status.error_embedded_estimate >= self.params.e_tol:
                S.status.restart = True

        super(Adaptivity, self).determine_restart(controller, S)
