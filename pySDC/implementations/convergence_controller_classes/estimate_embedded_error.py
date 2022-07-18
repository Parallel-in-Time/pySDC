import numpy as np

from pySDC.core.ConvergenceController import ConvergenceController
from pySDC.implementations.convergence_controller_classes.store_uold import StoreUOld


class EstimateEmbeddedError(ConvergenceController):
    '''
    The embedded error is obtained by computing two solutions of different accuracy and pretending the more accurate
    one is an exact solution from the point of view of the less accurate solution. In practice, we like to compute the
    solutions with different order methods, meaning that in SDC we can just subtract two consecutive sweeps, as long as
    you make sure your preconditioner is compatible, which you have to just try out...
    '''

    def setup(self, controller, params, description):
        return {'control_order': -80} | params

    def dependencies(self, controller, description):
        controller.add_convergence_controller(StoreUOld, {}, description=description)


class EstimateEmbeddedErrorNonMPI(EstimateEmbeddedError):

    def reset_global_variables_nonMPI(self, controller):
        self.e_em_last = 0.

    def post_iteration_processing(self, controller, S):
        """
        Compute embedded error estimate on the last node of each level
        In serial this is the local error, but in block Gauss-Seidel MSSDC this is a semi-global error in each block
        """
        if len(S.levels) > 1 and len(controller.MS) > 1:
            raise NotImplementedError('Embedded error estimate only works for serial multi-level or parallel single \
level')

        if S.status.iter > 1:
            for L in S.levels:
                # order rises by one between sweeps, making this so ridiculously easy
                temp = abs(L.uold[-1] - L.u[-1])
                L.status.error_embedded_estimate = max([abs(temp - self.e_em_last), np.finfo(float).eps])

            self.e_em_last = temp * 1.
