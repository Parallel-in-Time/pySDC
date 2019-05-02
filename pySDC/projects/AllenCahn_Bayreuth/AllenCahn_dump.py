import numpy as np
import json
from mpi4py import MPI
from pySDC.core.Hooks import hooks


class dump(hooks):

    def __init__(self):
        """
        Initialization of Allen-Cahn monitoring
        """
        super(dump, self).__init__()

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
        super(dump, self).pre_run(step, level_number)
        L = step.levels[0]

        # get space-communicator and data
        self.comm = L.prob.params.comm
        if self.comm is not None:
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.rank = 0
            self.size = 1

        # get real space values
        if L.prob.params.spectral:
            tmp = L.prob.fft.backward(L.u[0])
        else:
            tmp = L.u[0][:]

        # compute local offset for I/O
        nbytes_local = tmp.nbytes
        if self.comm is not None:
            nbytes_global = self.comm.allgather(nbytes_local)
        else:
            nbytes_global = [nbytes_local]
        local_offset = sum(nbytes_global[:self.rank])

        # dump initial data
        fname = f"./data/{L.prob.params.name}_{0:08d}"
        fh = MPI.File.Open(self.comm, fname + ".dat", self.amode)
        fh.Write_at_all(local_offset, tmp)
        fh.Close()

        # write json description
        if self.rank == 0 and step.status.slot == 0:
            json_obj = dict()
            json_obj['type'] = 'dataset'
            json_obj['datatype'] = str(tmp.dtype)
            json_obj['endian'] = str(tmp.dtype.byteorder)
            json_obj['time'] = L.time
            json_obj['space_comm_size'] = self.size
            json_obj['time_comm_size'] = step.status.time_size
            json_obj['shape'] = L.prob.params.nvars
            json_obj['elementsize'] = tmp.dtype.itemsize

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
        super(dump, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[0]

        # get real space values
        if L.prob.params.spectral:
            tmp = L.prob.fft.backward(L.uend)
        else:
            tmp = L.uend[:]

        # compute local offset for I/O
        nbytes_local = tmp.nbytes
        if self.comm is not None:
            nbytes_global = self.comm.allgather(nbytes_local)
        else:
            nbytes_global = [nbytes_local]
        local_offset = sum(nbytes_global[:self.rank])

        #  dump initial data
        fname = f"./data/{L.prob.params.name}_{self.time_step + step.status.slot:08d}"
        fh = MPI.File.Open(self.comm, fname + ".dat", self.amode)
        fh.Write_at_all(local_offset, tmp)
        fh.Close()

        # write json description
        if self.rank == 0:
            json_obj = dict()
            json_obj['type'] = 'dataset'
            json_obj['datatype'] = str(tmp.dtype)
            json_obj['endian'] = str(tmp.dtype.byteorder)
            json_obj['time'] = L.time + L.dt
            json_obj['space_comm_size'] = self.size
            json_obj['time_comm_size'] = step.status.time_size
            json_obj['shape'] = L.prob.params.nvars
            json_obj['elementsize'] = tmp.dtype.itemsize

            with open(fname + '.json', 'w') as fp:
                json.dump(json_obj, fp)

        # update step count
        self.time_step += step.status.time_size
