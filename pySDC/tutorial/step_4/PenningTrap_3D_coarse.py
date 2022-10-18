import numpy as np

from pySDC.implementations.problem_classes.PenningTrap_3D import penningtrap


class penningtrap_coarse(penningtrap):
    """
    Coarse level problem description class, will only overwrite what is needed
    """

    def eval_f(self, part, t):
        """
        Routine to compute the E and B fields (named f for consistency with the original PEPC version)

        Args:
            t: current time (not used here)
            part: the particles
        Returns:
            Fields for the particles (external only)
        """

        N = self.params.nparts

        Emat = np.diag([1, 1, -2])
        f = self.dtype_f(self.init, val=0.0)

        # only compute external forces here: O(N) instead of O(N*N)
        for n in range(N):
            f.elec[:, n] = self.params.omega_E**2 / (part.q[n] / part.m[n]) * np.dot(Emat, part.pos[:, n])
            f.magn[:, n] = self.params.omega_B * np.array([0, 0, 1])

        return f
