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

        self.work_counters['rhs']()  # count the evaluations, as the fine problem does
        Emat = np.diag([1, 1, -2])
        f = self.dtype_f(self.init, val=0.0)

        # only compute external forces here: O(N) instead of O(N*N)
        f.elec[:] = self.omega_E**2 / (part.q / part.m) * (Emat @ part.pos)
        f.magn[:] = self.omega_B * np.array([0, 0, 1])[:, None]

        return f
