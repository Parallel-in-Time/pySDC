import numpy as np

from pySDC.core.ConvergenceController import ConvergenceController
from pySDC.implementations.convergence_controller_classes.store_uold import StoreUOld


class EstimateEmbeddedError(ConvergenceController):

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

        for L in S.levels:
            # order rises by one between sweeps, making this so ridiculously easy
            temp = abs(L.uold[-1] - L.u[-1])
            L.status.error_embedded_estimate = max([abs(temp - self.e_em_last), np.finfo(float).eps])

        self.e_em_last = temp * 1.
