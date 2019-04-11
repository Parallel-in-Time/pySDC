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
        self.ndim = None

        self.comm = None
        self.rank = None
        self.size = None
        self.amode = MPI.MODE_WRONLY | MPI.MODE_CREATE
        self.fh = None
        self.json_obj = {}
        self.time_step = None
        self.fname = None

    def pre_run(self, step, level_number):
        """
        Overwrite standard pre run hook

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super(monitor, self).pre_run(step, level_number)
        L = step.levels[0]

        self.comm = L.prob.pm.comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.ndim = len(L.u[0].values.shape)
        c_local = np.count_nonzero(L.u[0].values > 0.0)
        if self.comm is not None:
            c_global = self.comm.allreduce(sendobj=c_local, op=MPI.SUM)
        else:
            c_global = c_local
        if self.ndim == 3:
            radius = (c_global / (np.pi * 4.0 / 3.0)) ** (1.0/3.0) * L.prob.dx
        elif self.ndim == 2:
            radius = np.sqrt(c_global / np.pi) * L.prob.dx
        else:
            raise NotImplementedError('Can use this only for 2 or 3D problems')

        self.init_radius = L.prob.params.radius

        if L.time == 0.0:
            self.add_to_stats(process=step.status.slot, time=L.time, level=-1, iter=step.status.iter,
                              sweep=L.status.sweep, type='computed_radius', value=radius)
            self.add_to_stats(process=step.status.slot, time=L.time, level=-1, iter=step.status.iter,
                              sweep=L.status.sweep, type='exact_radius', value=self.init_radius)

        self.time_step = 0

        # `todo: add initial condition dump

    def pre_step(self, step, level_number):

        super(monitor, self).pre_step(step, level_number)
        L = step.levels[0]

        time_step = self.time_step + step.status.slot

        self.fname = f"./data/{L.prob.params.name}_{time_step:08d}"
        self.fh = MPI.File.Open(self.comm, self.fname + ".dat", self.amode)

        if self.rank == 0:
            self.json_obj['type'] = 'dataset'
            self.json_obj['datatype'] = str(L.u[0].values.dtype)
            self.json_obj['endian'] = str(L.u[0].values.dtype.byteorder)
            self.json_obj['time'] = L.time
            self.json_obj['space_comm_size'] = self.size
            self.json_obj['time_comm_size'] = step.status.time_size
            self.json_obj['shape'] = L.prob.params.nvars
            self.json_obj['elementsize'] = L.u[0].values.dtype.itemsize

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

        c_local = np.count_nonzero(L.uend.values > 0.0)

        if self.comm is not None:
            c_global = self.comm.allreduce(sendobj=c_local, op=MPI.SUM)
        else:
            c_global = c_local
        if self.ndim == 3:
            radius = (c_global / (np.pi * 4.0 / 3.0)) ** (1.0 / 3.0) * L.prob.dx
        elif self.ndim == 2:
            radius = np.sqrt(c_global / np.pi) * L.prob.dx
        else:
            raise NotImplementedError('Can use this only for 2 or 3D problems')

        exact_radius = np.sqrt(max(self.init_radius ** 2 - 2.0 * (self.ndim - 1) * (L.time + L.dt), 0))

        self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=-1, iter=step.status.iter,
                          sweep=L.status.sweep, type='computed_radius', value=radius)
        self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=-1, iter=step.status.iter,
                          sweep=L.status.sweep, type='exact_radius', value=exact_radius)

        nbytes_local = L.uend.values.nbytes
        if self.comm is not None:
            nbytes_global = self.comm.allgather(nbytes_local)
        else:
            nbytes_global = [nbytes_local]

        # compute local offset and write
        local_offset = sum(nbytes_global[:self.rank])
        self.fh.Write_at_all(local_offset, L.uend.values)
        # update offset by adding space-time block

        self.time_step += step.status.time_size
        self.fh.Close()

        if self.rank == 0:
            with open(self.fname + '.json', 'w') as fp:
                json.dump(self.json_obj, fp)
