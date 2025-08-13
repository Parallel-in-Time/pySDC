import numpy as np

from pySDC.projects.parallelSDC.linearized_implicit_fixed_parallel import linearized_implicit_fixed_parallel


class linearized_implicit_fixed_parallel_prec(linearized_implicit_fixed_parallel):
    """
    Custom sweeper class, implements Sweeper.py

    Attributes:
        D: eigenvalues of the QI
    """

    def __init__(self, params, level):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
            level (pySDC.Level.level): the level that uses this sweeper
        """

        if 'fixed_time_in_jacobian' not in params:
            params['fixed_time_in_jacobian'] = 0

        super().__init__(params, level)

        assert self.params.fixed_time_in_jacobian in range(self.coll.num_nodes + 1), (
            "ERROR: fixed_time_in_jacobian is too small or too large, got %s" % self.params.fixed_time_in_jacobian
        )

        self.D, self.V = np.linalg.eig(self.QI[1:, 1:])
        self.Vi = np.linalg.inv(self.V)
