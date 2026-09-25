import numpy as np

from pySDC.core.hooks import Hooks


class AllenCahnMonitor(Hooks):
    r"""
    Track the shrinking circle (or sphere) of an Allen-Cahn run.

    Under mean curvature flow a blob of initial radius :math:`R_0` obeys
    :math:`R(t)^2 = R_0^2 - 2 (d - 1) t`, so comparing the measured radius against that is the
    standard diagnostic for these problems. This hook records ``computed_radius``,
    ``exact_radius``, ``computed_volume`` and ``exact_volume`` at :math:`t = 0` and after every
    step, plus ``interface_width`` where that is well defined.

    The volume is the number of cells in the high phase times the cell volume. Counting is used
    rather than integrating the field, even though pySDC's phase fields run from :math:`0` to
    :math:`1` and so integrate straight to a volume: the integral also picks up the diffuse
    interface, which biases it by :math:`O(\varepsilon)` no matter how fine the mesh, whereas
    counting is consistent and its :math:`O(\Delta x)` bias refines away.

    Attributes
    ----------
    phase_thresh : float
        Cells above this count towards the high phase. Taken from the problem when it says, and
        :math:`0.5` -- the midpoint of the two wells -- otherwise.
    """

    default_phase_thresh = 0.5

    def __init__(self):
        super().__init__()

        self.init_radius = None
        self.ndim = None
        self.phase_thresh = self.default_phase_thresh

    @staticmethod
    def get_real_space(L, u):
        """Undo the transform if the problem carries its solution in spectral space."""
        return L.prob.fft.backward(u) if getattr(L.prob, 'spectral', False) else u[:]

    def get_volume(self, L, u):
        """Volume of the high phase, summed over the space communicator if there is one."""
        count = float(np.count_nonzero(self.get_real_space(L, u) > self.phase_thresh))

        comm = getattr(L.prob, 'comm', None)
        if comm is not None:
            from mpi4py import MPI

            count = comm.allreduce(sendobj=count, op=MPI.SUM)

        return count * L.prob.dx**self.ndim

    def radius_from_volume(self, vol):
        """Radius of the ball of this volume."""
        if self.ndim == 2:
            return np.sqrt(vol / np.pi)
        elif self.ndim == 3:
            return (vol / (np.pi * 4.0 / 3.0)) ** (1.0 / 3.0)
        raise NotImplementedError(f'Can only monitor 2D and 3D problems, got {self.ndim}D')

    def exact_radius_squared(self, t):
        r"""Mean curvature flow shrinks the blob as :math:`R(t)^2 = R_0^2 - 2 (d - 1) t`."""
        return max(self.init_radius**2 - 2.0 * (self.ndim - 1) * t, 0)

    def exact_radius(self, t):
        return np.sqrt(self.exact_radius_squared(t))

    def exact_volume(self, t):
        r2 = self.exact_radius_squared(t)
        return np.pi * r2 if self.ndim == 2 else np.pi * 4.0 / 3.0 * r2**1.5

    def measures_interface_width(self, L):
        """Only a 2D field held whole on this rank can be cut across to measure the interface."""
        comm = getattr(L.prob, 'comm', None)
        return self.ndim == 2 and (comm is None or comm.Get_size() == 1)

    def get_interface_width(self, L, u):
        """Width of the transition, in units of epsilon, along a cut through the middle."""
        n = L.prob.init[0][0]
        rows1 = np.where(u[n // 2, : n // 2] > 0.005)
        rows2 = np.where(u[n // 2, : n // 2] < 0.995)

        return (rows2[0][-1] - rows1[0][0]) * L.prob.dx / L.prob.eps

    def get_diagnostics(self, L, u, t):
        """Everything worth recording about ``u``, as a dict of stats entries."""
        vol = self.get_volume(L, u)

        diagnostics = {
            'computed_radius': self.radius_from_volume(vol),
            'exact_radius': self.exact_radius(t),
            'computed_volume': vol,
            'exact_volume': self.exact_volume(t),
        }

        if self.measures_interface_width(L):
            diagnostics['interface_width'] = self.get_interface_width(L, self.get_real_space(L, u))

        return diagnostics

    def record(self, step, L, t, diagnostics):
        for key, value in diagnostics.items():
            self.add_to_stats(
                process=step.status.slot,
                time=t,
                level=-1,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type=key,
                value=value,
            )

    def pre_run(self, step, level_number):
        super().pre_run(step, level_number)
        L = step.levels[0]

        self.init_radius = L.prob.radius
        self.phase_thresh = getattr(L.prob, 'phase_thresh', self.default_phase_thresh)
        self.ndim = len(self.get_real_space(L, L.u[0]).shape)

        if L.time == 0.0:
            self.record(step, L, L.time, self.get_diagnostics(L, L.u[0], 0.0))

    def post_step(self, step, level_number):
        super().post_step(step, level_number)
        L = step.levels[0]

        self.record(step, L, L.time + L.dt, self.get_diagnostics(L, L.uend, L.time + L.dt))
