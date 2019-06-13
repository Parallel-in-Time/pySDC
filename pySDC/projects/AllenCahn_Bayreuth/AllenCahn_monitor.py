import numpy as np
import json
from mpi4py import MPI
from pySDC.core.Hooks import hooks


class monitor(hooks):

    def __init__(self):
        """
        Initialization of Allen-Cahn monitoring
        """
        super(monitor, self).__init__()

        self.init_radius = None
        self.init_vol = None
        self.ndim = None
        self.corr_rad = None
        self.corr_vol = None

        self.comm = None

    def pre_run(self, step, level_number):
        """
        Overwrite standard pre run hook

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super(monitor, self).pre_run(step, level_number)
        L = step.levels[0]

        # get space-communicator and data
        self.comm = L.prob.params.comm

        # get real space values
        if L.prob.params.spectral:
            tmp = L.prob.fft.backward(L.u[0])
        else:
            tmp = L.u[0][:]
        self.ndim = len(tmp.shape)

        # compute numerical radius and volume
        # c_local = np.count_nonzero(tmp >= 0.5)
        c_local = float(tmp[:].sum())
        if self.comm is not None:
            c_global = self.comm.allreduce(sendobj=c_local, op=MPI.SUM)
        else:
            c_global = c_local
        if self.ndim == 3:
            vol = c_global * L.prob.dx ** 3
            radius = (vol / (np.pi * 4.0 / 3.0)) ** (1.0 / 3.0)
            self.init_vol = np.pi * 4.0 / 3.0 * L.prob.params.radius ** 3
        elif self.ndim == 2:
            vol = c_global * L.prob.dx ** 2
            radius = np.sqrt(vol / np.pi)
            self.init_vol = np.pi * L.prob.params.radius ** 2
        else:
            raise NotImplementedError('Can use this only for 2 or 3D problems')

        self.init_radius = L.prob.params.radius
        self.corr_rad = self.init_radius / radius
        self.corr_vol = self.init_vol / vol
        radius *= self.corr_rad
        vol *= self.corr_vol

        # write to stats
        if L.time == 0.0:
            self.add_to_stats(process=step.status.slot, time=L.time, level=-1, iter=step.status.iter,
                              sweep=L.status.sweep, type='computed_radius', value=radius)
            self.add_to_stats(process=step.status.slot, time=L.time, level=-1, iter=step.status.iter,
                              sweep=L.status.sweep, type='exact_radius', value=self.init_radius)
            self.add_to_stats(process=step.status.slot, time=L.time, level=-1, iter=step.status.iter,
                              sweep=L.status.sweep, type='computed_volume', value=vol)
            self.add_to_stats(process=step.status.slot, time=L.time, level=-1, iter=step.status.iter,
                              sweep=L.status.sweep, type='exact_volume', value=self.init_vol)

    def post_step(self, step, level_number):
        """
        Overwrite standard post step hook

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super(monitor, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[0]

        # get real space values
        if L.prob.params.spectral:
            tmp = L.prob.fft.backward(L.uend)
        else:
            tmp = L.uend[:]

        # compute numerical radius and volume
        # c_local = np.count_nonzero(tmp >= 0.5)
        # c_local = float(tmp[tmp > 2 * L.prob.params.eps].sum())
        c_local = float(tmp[:].sum())
        if self.comm is not None:
            c_global = self.comm.allreduce(sendobj=c_local, op=MPI.SUM)
        else:
            c_global = c_local

        if self.ndim == 3:
            vol = c_global * L.prob.dx ** 3
            radius = (vol / (np.pi * 4.0 / 3.0)) ** (1.0 / 3.0)
            exact_vol = np.pi * 4.0 / 3.0 * (max(self.init_radius ** 2 - 4.0 * (L.time + L.dt), 0)) ** (3.0 / 2.0)
            exact_radius = (exact_vol / (np.pi * 4.0 / 3.0)) ** (1.0 / 3.0)
        elif self.ndim == 2:
            vol = c_global * L.prob.dx ** 2
            radius = np.sqrt(vol / np.pi)
            exact_vol = np.pi * max(self.init_radius ** 2 - 2.0 * (L.time + L.dt), 0)
            exact_radius = np.sqrt(exact_vol / np.pi)
        else:
            raise NotImplementedError('Can use this only for 2 or 3D problems')

        radius *= self.corr_rad
        vol *= self.corr_vol

        # write to stats
        self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=-1, iter=step.status.iter,
                          sweep=L.status.sweep, type='computed_radius', value=radius)
        self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=-1, iter=step.status.iter,
                          sweep=L.status.sweep, type='exact_radius', value=exact_radius)
        self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=-1, iter=step.status.iter,
                          sweep=L.status.sweep, type='computed_volume', value=vol)
        self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=-1, iter=step.status.iter,
                          sweep=L.status.sweep, type='exact_volume', value=exact_vol)
