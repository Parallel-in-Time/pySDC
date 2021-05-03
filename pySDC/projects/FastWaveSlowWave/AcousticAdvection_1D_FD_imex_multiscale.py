import numpy as np

from pySDC.implementations.problem_classes.AcousticAdvection_1D_FD_imex import acoustic_1d_imex


# noinspection PyUnusedLocal
class acoustic_1d_imex_multiscale(acoustic_1d_imex):
    """
    Example implementing the one-dimensional IMEX acoustic-advection with multiscale initial values
    """

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        sigma_0 = 0.1
        k = 7.0 * 2.0 * np.pi
        x_0 = 0.75
        x_1 = 0.25

        ms = 1.0

        me = self.dtype_u(self.init)
        me[0, :] = np.exp(-np.square(self.mesh - x_0 - self.params.cs * t) / (sigma_0 * sigma_0)) + \
            ms * np.exp(-np.square(self.mesh - x_1 - self.params.cs * t) / (sigma_0 * sigma_0)) * \
            np.cos(k * (self.mesh - self.params.cs * t) / sigma_0)
        me[1, :] = me[0, :]

        return me
