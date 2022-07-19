from pySDC.core.ConvergenceController import ConvergenceController


class StoreUOld(ConvergenceController):

    def setup(self, controller, params, description):
        return {'control_order': +90, **params}

    def post_iteration_processing(self, controller, S):
        '''
        Throw away the final sweep to match the error estimates.
        '''
        for L in S.levels:
            L.uold[:] = L.u[:]
