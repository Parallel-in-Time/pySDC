import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg import spsolve

from pySDC.implementations.problem_classes.GeneralizedFisher_1D_FD_implicit import generalized_fisher


# noinspection PyUnusedLocal
class generalized_fisher_jac(generalized_fisher):

    def eval_jacobian(self, u):
        """
        Evaluation of the Jacobian of the right-hand side

        Args:
            u: space values

        Returns:
            Jacobian matrix
        """

        # noinspection PyTypeChecker
        dfdu = self.A[1:-1, 1:-1] + sp.diags(self.params.lambda0 ** 2 - self.params.lambda0 ** 2 *
                                             (self.params.nu + 1) * u.values ** self.params.nu, offsets=0)
        # print(type(dfdu))
        return dfdu
