import numpy as np

from pySDC.core.hooks import Hooks


class AllenCahnMonitor(Hooks):
    r"""
    Track the shrinking circle (or sphere) of an Allen-Cahn run.

    Under mean curvature flow a blob of initial radius :math:`R_0` obeys
    :math:`R(t)^2 = R_0^2 - 2 (d - 1) t`, so comparing the measured radius against that is the
    standard diagnostic for these problems. This hook records ``computed_radius``,
    ``exact_radius``, ``computed_volume`` and ``exact_volume`` at :math:`t = 0` and after every
    step.

    Measuring the volume takes one of two forms, because pySDC spells Allen-Cahn two ways (see
    issue #434). With wells at :math:`0` and :math:`1` the field *is* the indicator of the high
    phase, so summing it gives the volume directly and resolves the diffuse interface. With wells
    at :math:`\pm 1` that sum would return the difference of the two phases instead, and the
    volume has to come from counting cells above a threshold. ``phase_thresh`` picks between them.

    Both estimators are biased by the diffuse interface. Setting ``calibrate`` divides that bias
    out by rescaling everything so that :math:`t = 0` reproduces the problem's own ``radius``.

    Attributes
    ----------
    phase_thresh : float or None
        ``None`` integrates the field, for a problem with wells at :math:`0` and :math:`1`.
        A float counts the cells above it, for a problem with wells at :math:`\pm 1`.
    calibrate : bool
        Rescale radius and volume so that the measurement at :math:`t = 0` is exact.
    """

    phase_thresh = None
    calibrate = False

    def __init__(self):
        super().__init__()

        self.init_radius = None
        self.ndim = None
        self.corr_rad = 1.0
        self.corr_vol = 1.0

    @staticmethod
    def get_real_space(L, u):
        """Undo the transform if the problem carries its solution in spectral space."""
        return L.prob.fft.backward(u) if getattr(L.prob, 'spectral', False) else u[:]

    @classmethod
    def count_high_phase(cls, u):
        """Cells occupied by the high phase, in units of cells."""
        if cls.phase_thresh is None:
            return float(u[:].sum())
        return float(np.count_nonzero(u > cls.phase_thresh))

    def get_volume(self, L, u):
        """Volume of the high phase, summed over the space communicator if there is one."""
        count = self.count_high_phase(self.get_real_space(L, u))

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

    def get_diagnostics(self, L, u, t):
        """Everything worth recording about ``u``, as a dict of stats entries."""
        vol = self.get_volume(L, u)
        exact_vol = self.exact_volume(t)

        return {
            'computed_radius': self.radius_from_volume(vol) * self.corr_rad,
            'exact_radius': self.exact_radius(t),
            'computed_volume': vol * self.corr_vol,
            'exact_volume': exact_vol,
        }

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
        self.ndim = len(self.get_real_space(L, L.u[0]).shape)

        if self.calibrate:
            vol = self.get_volume(L, L.u[0])
            self.corr_rad = self.init_radius / self.radius_from_volume(vol)
            self.corr_vol = self.exact_volume(0.0) / vol

        if L.time == 0.0:
            self.record(step, L, L.time, self.get_diagnostics(L, L.u[0], 0.0))

    def post_step(self, step, level_number):
        super().post_step(step, level_number)
        L = step.levels[0]

        self.record(step, L, L.time + L.dt, self.get_diagnostics(L, L.uend, L.time + L.dt))


class AllenCahnMonitorWithInterface(AllenCahnMonitor):
    """Adds the interface width, which only makes sense for a 2D field with wells at +-1."""

    phase_thresh = 0.0

    @staticmethod
    def get_interface_width(u, L):
        n = L.prob.init[0][0]
        rows1 = np.where(u[n // 2, : n // 2] > -0.99)
        rows2 = np.where(u[n // 2, : n // 2] < 0.99)

        return (rows2[0][-1] - rows1[0][0]) * L.prob.dx / L.prob.eps

    def get_diagnostics(self, L, u, t):
        diagnostics = super().get_diagnostics(L, u, t)
        diagnostics['interface_width'] = self.get_interface_width(u, L)
        return diagnostics
