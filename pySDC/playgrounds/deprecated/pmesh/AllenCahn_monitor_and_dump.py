import numpy as np
import json
from mpi4py import MPI
from pySDC.core.Hooks import hooks


class monitor_and_dump(hooks):
    def __init__(self):
        """
        Initialization of Allen-Cahn monitoring
        """
        super(monitor_and_dump, self).__init__()

        self.init_radius = None
        self.ndim = None

        self.comm = None
        self.rank = None
        self.size = None
        self.amode = MPI.MODE_WRONLY | MPI.MODE_CREATE
        self.time_step = None

    def pre_run(self, step, level_number):
        """
        Overwrite standard pre run hook

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super(monitor_and_dump, self).pre_run(step, level_number)
        L = step.levels[0]

        # get space-communicator and data
        self.comm = L.prob.pm.comm
        if self.comm is not None:
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.rank = 0
            self.size = 1

        # compute numerical radius
        self.ndim = len(L.u[0].values.shape)
        v_local = L.u[0].values[L.u[0].values > 2 * L.prob.params.eps].sum()
        if self.comm is not None:
            v_global = self.comm.allreduce(sendobj=v_local, op=MPI.SUM)
        else:
            v_global = v_local
        if self.ndim == 3:
            radius = (v_global / (np.pi * 4.0 / 3.0)) ** (1.0 / 3.0) * L.prob.dx
        elif self.ndim == 2:
            radius = np.sqrt(v_global / np.pi) * L.prob.dx
        else:
            raise NotImplementedError('Can use this only for 2 or 3D problems')

        # c_local = np.count_nonzero(L.u[0].values > 0.5)
        # if self.comm is not None:
        #     c_global = self.comm.allreduce(sendobj=c_local, op=MPI.SUM)
        # else:
        #     c_global = c_local
        # if self.ndim == 3:
        #     radius = (c_global / (np.pi * 4.0 / 3.0)) ** (1.0/3.0) * L.prob.dx
        # elif self.ndim == 2:
        #     radius = np.sqrt(c_global / np.pi) * L.prob.dx
        # else:
        #     raise NotImplementedError('Can use this only for 2 or 3D problems')

        self.init_radius = L.prob.params.radius

        # write to stats
        if L.time == 0.0:
            self.add_to_stats(
                process=step.status.slot,
                time=L.time,
                level=-1,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type='computed_radius',
                value=radius,
            )
            self.add_to_stats(
                process=step.status.slot,
                time=L.time,
                level=-1,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type='exact_radius',
                value=self.init_radius,
            )

        # compute local offset for I/O
        nbytes_local = L.u[0].values.nbytes
        if self.comm is not None:
            nbytes_global = self.comm.allgather(nbytes_local)
        else:
            nbytes_global = [nbytes_local]
        local_offset = sum(nbytes_global[: self.rank])

        # dump initial data
        fname = f"./data/{L.prob.params.name}_{0:08d}"
        fh = MPI.File.Open(self.comm, fname + ".dat", self.amode)
        fh.Write_at_all(local_offset, L.u[0].values)
        fh.Close()

        # write json description
        if self.rank == 0 and step.status.slot == 0:
            json_obj = dict()
            json_obj['type'] = 'dataset'
            json_obj['datatype'] = str(L.u[0].values.dtype)
            json_obj['endian'] = str(L.u[0].values.dtype.byteorder)
            json_obj['time'] = L.time
            json_obj['space_comm_size'] = self.size
            json_obj['time_comm_size'] = step.status.time_size
            json_obj['shape'] = L.prob.params.nvars
            json_obj['elementsize'] = L.u[0].values.dtype.itemsize

            with open(fname + '.json', 'w') as fp:
                json.dump(json_obj, fp)

        # set step count
        self.time_step = 1

    def post_step(self, step, level_number):
        """
        Overwrite standard post step hook

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super(monitor_and_dump, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[0]

        # compute numerical radius
        # v_local = np.sum(L.uend.values)
        v_local = L.uend.values[L.uend.values > 2 * L.prob.params.eps].sum()
        if self.comm is not None:
            v_global = self.comm.allreduce(sendobj=v_local, op=MPI.SUM)
        else:
            v_global = v_local
        if self.ndim == 3:
            radius = (v_global / (np.pi * 4.0 / 3.0)) ** (1.0 / 3.0) * L.prob.dx
        elif self.ndim == 2:
            radius = np.sqrt(v_global / np.pi) * L.prob.dx
        else:
            raise NotImplementedError('Can use this only for 2 or 3D problems')

        # c_local = np.count_nonzero(L.uend.values > 0.5)
        # if self.comm is not None:
        #     c_global = self.comm.allreduce(sendobj=c_local, op=MPI.SUM)
        # else:
        #     c_global = c_local
        # if self.ndim == 3:
        #     radius = (c_global / (np.pi * 4.0 / 3.0)) ** (1.0 / 3.0) * L.prob.dx
        # elif self.ndim == 2:
        #     radius = np.sqrt(c_global / np.pi) * L.prob.dx
        # else:
        #     raise NotImplementedError('Can use this only for 2 or 3D problems')

        # compute exact radius
        exact_radius = np.sqrt(max(self.init_radius**2 - 2.0 * (self.ndim - 1) * (L.time + L.dt), 0))

        # write to stats
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=-1,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='computed_radius',
            value=radius,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=-1,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='exact_radius',
            value=exact_radius,
        )

        # compute local offset for I/O
        nbytes_local = L.uend.values.nbytes
        if self.comm is not None:
            nbytes_global = self.comm.allgather(nbytes_local)
        else:
            nbytes_global = [nbytes_local]
        local_offset = sum(nbytes_global[: self.rank])

        #  dump initial data
        fname = f"./data/{L.prob.params.name}_{self.time_step + step.status.slot:08d}"
        fh = MPI.File.Open(self.comm, fname + ".dat", self.amode)
        fh.Write_at_all(local_offset, L.uend.values)
        fh.Close()

        # write json description
        if self.rank == 0:
            json_obj = dict()
            json_obj['type'] = 'dataset'
            json_obj['datatype'] = str(L.uend.values.dtype)
            json_obj['endian'] = str(L.uend.values.dtype.byteorder)
            json_obj['time'] = L.time + L.dt
            json_obj['space_comm_size'] = self.size
            json_obj['time_comm_size'] = step.status.time_size
            json_obj['shape'] = L.prob.params.nvars
            json_obj['elementsize'] = L.uend.values.dtype.itemsize

            with open(fname + '.json', 'w') as fp:
                json.dump(json_obj, fp)

        # update step count
        self.time_step += step.status.time_size
