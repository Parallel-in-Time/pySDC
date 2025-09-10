#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions building 2D/3D Rayleigh-Benard Convection problems with Dedalus
"""
import os
import socket
import glob
from datetime import datetime
from time import sleep

import numpy as np
import dedalus.public as d3
from mpi4py import MPI
import h5py
import scipy.optimize as sco

from qmat.lagrange import LagrangeApproximation
from pySDC.helpers.fieldsIO import Rectilinear
from pySDC.helpers.blocks import BlockDecomposition
from pySDC.playgrounds.dedalus.timestepper import SDCIMEX, SDCIMEX_MPI, SDCIMEX_MPI2

COMM_WORLD = MPI.COMM_WORLD
MPI_SIZE = COMM_WORLD.Get_size()
MPI_RANK = COMM_WORLD.Get_rank()


class RBCProblem2D():

    def __init__(self, Rayleigh=1e7, Prandtl=1,
                 resFactor=1, aspectRatio=4, meshRatio=1,
                 sComm=COMM_WORLD, mpiBlocks=None, writeSpaceDistr=False,
                 initFields=None, seed=999):

        self.Rayleigh, self.Prandtl = Rayleigh, Prandtl
        self.resFactor = resFactor

        self.infos = {
            "Ra": Rayleigh,
            "Pr": Prandtl,
            }

        self.buildGrid(aspectRatio, meshRatio, sComm, mpiBlocks)
        if writeSpaceDistr: self.writeSpaceDistr()
        self.buildProblem()
        self.initFields(initFields, seed)


    def buildGrid(self, aspectRatio, meshRatio, sComm, mpiBlocks):
        baseSize = 64
        Lx, Lz = aspectRatio, 1
        Nx = int(baseSize*aspectRatio*meshRatio*self.resFactor)
        Nz = int(baseSize*self.resFactor)
        dealias = 3/2
        dtype = np.float64

        coords = d3.CartesianCoordinates('x', 'z')
        dist = d3.Distributor(coords, dtype=dtype, mesh=mpiBlocks, comm=sComm)
        xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
        zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
        self.bases, self.axes = (xbasis, zbasis), "xz"
        self.coords, self.dist, self.sComm = coords, dist, sComm

        self.infos.update(Nx=Nx, Nz=Nz, Lx=Lx, Lz=Lz, mpiBlocks=mpiBlocks)
        self.mpiBlocks = mpiBlocks


    @property
    def Lx(self):
        return self.bases[0].bounds[-1]

    @property
    def Nx(self):
        return self.bases[0].size

    @property
    def Lz(self):
        return self.bases[-1].bounds[-1]

    @property
    def Nz(self):
        return self.bases[-1].size


    def printSpaceDistr(self):
        grids = self.grids
        x, z = [grids[axis] for axis in self.axes]
        MPI.COMM_WORLD.Barrier()
        sleep(0.01*MPI_RANK)
        print(f"Rank {MPI_RANK}({MPI_SIZE}) :\n"
              f"\tx: {x.shape}, [{x.min(initial=np.inf)}, {x.max(initial=-np.inf)}]\n"
              f"\tz: {z.shape}, [{z.min(initial=np.inf)}, {z.max(initial=-np.inf)}]\n"
              f"\tcpu: {os.sched_getaffinity(0)}, on {socket.gethostname()}", flush=True)
        MPI.COMM_WORLD.Barrier()


    @property
    def grids(self):
        return  {c: b.local_grid(self.dist, scale=1)
                 for c, b in zip(self.axes, self.bases)}


    @property
    def dim(self):
        return len(self.bases)


    def buildProblem(self):
        dist, coords, bases = self.dist, self.coords, self.bases

        p = dist.Field(name='p', bases=bases)
        b = dist.Field(name='b', bases=bases)
        u = dist.VectorField(coords, name='u', bases=bases)
        tau_p = dist.Field(name='tau_p')
        tau_b1 = dist.Field(name='tau_b1', bases=bases[:-1])
        tau_b2 = dist.Field(name='tau_b2', bases=bases[:-1])
        tau_u1 = dist.VectorField(coords, name='tau_u1', bases=bases[:-1])
        tau_u2 = dist.VectorField(coords, name='tau_u2', bases=bases[:-1])

        self.fields = {name: field for field, name in [
            (b, "buoyancy"), (p, "pressure"), (u, "velocity"),
            *[(f, f.name) for f in [tau_p, tau_b1, tau_b2, tau_u1, tau_u2]]
            ]}

        Rayleigh, Prandtl, Lz = self.Rayleigh, self.Prandtl, self.Lz

        # -- operators and substitutions
        kappa = (Rayleigh * Prandtl)**(-1/2)
        nu = (Rayleigh / Prandtl)**(-1/2)
        *_, z = dist.local_grids(*self.bases)
        *_, ez = coords.unit_vector_fields(dist)
        lift_basis = self.bases[-1].derivative_basis(1)
        lift = lambda A: d3.Lift(A, lift_basis, -1)
        grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
        grad_b = d3.grad(b) + ez*lift(tau_b1) # First-order reduction

        self.z = z

        # -- build problem
        # first-order form: "div(f)" becomes "trace(grad_f)"
        # first-order form: "lap(f)" becomes "div(grad_f)"
        problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
        problem.add_equation("trace(grad_u) + tau_p = 0")
        problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)")
        problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)")
        problem.add_equation("b(z=0) = Lz")
        problem.add_equation("u(z=0) = 0")
        problem.add_equation("b(z=Lz) = 0")
        problem.add_equation("u(z=Lz) = 0")
        problem.add_equation("integ(p) = 0") # Pressure gauge

        self.problem = problem


    def initFields(self, initFields, seed):
        b, u = self.fields["buoyancy"], self.fields["velocity"]
        z, Lz = self.z, self.Lz

        if initFields is None:  # linear buoyancy with random noise
            if MPI_RANK == 0: print(" -- generating randomly perturbed initial field")
            b.fill_random('g', seed=seed, distribution='normal', scale=1e-3) # Random noise
            b['g'] *= z * (Lz - z) # Damp noise at walls
            b['g'] += Lz - z # Add linear background
        else:
            if type(initFields) == h5py._hl.group.Group:
                if MPI_RANK == 0: print(" -- reading field from HDF5 file")
                for name, field in self.fields.items():
                    localSlices = (slice(None),) * len(field.tensorsig) \
                        + self.dist.grid_layout.slices(field.domain, field.scales)
                    try:
                        field['g'] = initFields[name][-1][localSlices]
                    except KeyError:
                        # field not present in file, put zeros instead
                        field['g'] = 0
            elif type(initFields) == Rectilinear:
                if MPI_RANK == 0: print(" -- reading field from pySDC file")
                sFields = {
                    "buoyancy": self.dim,
                    "pressure": self.dim+1,
                    "velocity": slice(self.dim)
                    }
                slices = self.dist.grid_layout.slices(u.domain, u.scales)
                Rectilinear.setupMPI(
                    self.sComm,
                    [s.start for s in slices], [s.stop-s.start for s in slices]
                    )
                _, uInit = initFields.readField(-1)
                for name, field in self.fields.items():
                    try:
                        field['g'] = uInit[sFields[name]]
                    except KeyError:
                        # field not present in file, put zeros instead
                        field['g'] = 0
        if MPI_RANK == 0: print(" -- done !")
        self.fields0 = {name: field.copy() for name, field in self.fields.items()}

    @classmethod
    def runSimulation(cls, dirName, tEnd, baseDt, tBeg=0, logEvery=100,
                      dtWrite=None, writeVort=False, writeTau=False,
                      timeScheme="RK443", timeParallel=False, groupTimeProcs=False,
                      **pParams):

        if timeScheme == "RK443":
            timeStepper = d3.RK443
        elif timeScheme == "RK111":
            timeStepper = d3.RK111
        elif timeScheme == "RK222":
            timeStepper = d3.RK222
        elif timeScheme == "SDC":
            timeStepper = SDCIMEX
        else:
            raise NotImplementedError(f"{timeStepper=}")

        if timeParallel:
            assert timeScheme == "SDC", "need timeScheme=SDC for timeParallel"
            _, sComm, _ = SDCIMEX_MPI.initSpaceTimeComms(groupTime=groupTimeProcs)
            pParams.update(sComm=sComm)
            if timeParallel == "MPI":
                timeScheme = SDCIMEX_MPI
            elif timeParallel == "MPI2":
                timeScheme = SDCIMEX_MPI2
            else:
                raise NotImplementedError(f"{timeParallel=}")

        p = cls(**pParams)

        dt = baseDt/p.resFactor
        nSteps = round(float(tEnd-tBeg)/dt, ndigits=3)
        if float(tEnd-tBeg) != round(nSteps*dt, ndigits=3):
            raise ValueError(f"{tEnd=} is not divisible by timestep {dt=} ({nSteps=})")
        nSteps = int(nSteps)
        p.infos.update(tEnd=tEnd, dt=dt, nSteps=nSteps)

        if os.path.isfile(f"{dirName}/01_finalized.txt"):
            if MPI_RANK == 0:
                print(" -- simulation already finalized, skipping !")
            return p
        os.makedirs(dirName, exist_ok=True)
        p.infos.update(dirName=dirName)

        # Solver
        if MPI_RANK == 0: print(" -- building dedalus solver ...")
        solver = p.problem.build_solver(timeStepper)
        solver.sim_time = tBeg
        solver.stop_sim_time = tEnd
        if MPI_RANK == 0: print(" -- finished building dedalus solver")

        # Fields IO
        if dtWrite:
            if MPI_RANK == 0: print(" -- setup fields IO")
            iterWrite = dtWrite/dt
            if int(iterWrite) != round(iterWrite, ndigits=3):
                raise ValueError(f"{dtWrite=} is not divisible by {dt=} ({iterWrite=})")
            iterWrite = int(iterWrite)
            snapshots = solver.evaluator.add_file_handler(
                dirName, sim_dt=dtWrite, max_writes=tEnd/dt)
            for name in ["velocity", "buoyancy", "pressure"]:
                snapshots.add_task(p.fields[name], name=name)
            if writeTau:
                for name in ["tau_p", "tau_b1", "tau_b2", "tau_u1", "tau_u2"]:
                    snapshots.add_task(p.fields[name], name=name)
            if writeVort:
                u = p.fields["velocity"]
                snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
            if MPI_RANK == 0: print(" -- done !")


        if MPI_RANK == 0:
            with open(f"{dirName}/00_infoSimu.txt", "w") as f:
                for key, val in p.infos.items():
                    if type(val) == float:
                        f.write(f"{key} : {val:1.2e}\n")
                    else:
                        f.write(f"{key} : {val}\n")

        def log(msg):
            if MPI_RANK == 0:
                with open(f"{dirName}/simu.log", "a") as f:
                    f.write(f"{dirName} -- ")
                    f.write(datetime.now().strftime("%d/%m/%Y  %H:%M:%S"))
                    f.write(f", MPI rank {MPI_RANK} ({MPI_SIZE})")
                    f.write(f" : {msg}\n")

        # Main loop
        if nSteps == 0:
            return p
        try:
            log('Starting main loop')
            solver.step(dt)   # don't count first time-step in timings
            log('Finished first time-step')
            t0 = MPI.Wtime()
            for _ in range(nSteps): # need to do one more step to write last solution ...
                solver.step(dt)
                if (solver.iteration) % logEvery == 0:
                    iStep = solver.iteration
                    simTime = float(solver.sim_time)
                    log(f'{iStep=}, {simTime=}, {dt=}')
            t1 = MPI.Wtime()
            p.infos.update(
                tComp=t1-t0,
                MPI_SIZE=MPI_SIZE,
                MPI_BLOCKS=p.mpiBlocks,
                tCompAll=(t1-t0)*MPI_SIZE)
            log('End of simulation')
            if MPI_RANK == 0:
                with open(f"{dirName}/01_finalized.txt", "w") as f:
                    f.write("Done !")
        except:
            log('Exception raised, triggering end of main loop.')
            raise
        finally:
            solver.log_stats()

        return p


class RBCProblem3D(RBCProblem2D):

    def __init__(self, Rayleigh=1e5, Prandtl=0.7,
                 resFactor=1, aspectRatio=4, meshRatio=0.5,
                 sComm=COMM_WORLD, mpiBlocks=None, writeSpaceDistr=False,
                 initFields=None, seed=999):
        super().__init__(
            Rayleigh, Prandtl,
            resFactor, aspectRatio, meshRatio,
            sComm, mpiBlocks, writeSpaceDistr,
            initFields, seed)

    def buildGrid(self, aspectRatio, meshRatio, sComm, mpiBlocks):
        baseSize = 32

        Lx, Ly, Lz = int(aspectRatio), int(aspectRatio), 1
        Nx = Ny = int(baseSize*aspectRatio*meshRatio*self.resFactor)
        Nz = int(baseSize*self.resFactor)
        dealias = 3/2
        dtype = np.float64

        if mpiBlocks is None:
            nProcs = sComm.Get_size()
            if Nz // nProcs >= 2:
                mpiBlocks = [1, nProcs]
            else:
                blocks = BlockDecomposition(nProcs, [Ny, Nz])
                mpiBlocks = blocks.nBlocks[-1::-1]
        if MPI_RANK == 0: print(f" -- {mpiBlocks = }")

        coords = d3.CartesianCoordinates('x', 'y', 'z')
        dist = d3.Distributor(coords, dtype=dtype, mesh=mpiBlocks, comm=sComm)
        xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
        ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)
        zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
        self.bases, self.axes = (xbasis, ybasis, zbasis), "xyz"
        self.coords, self.dist, self.sComm = coords, dist, sComm

        self.infos.update(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz)
        self.mpiBlocks = mpiBlocks

    @property
    def Ly(self):
        return self.bases[1].bounds[-1]

    @property
    def Ny(self):
        return self.bases[1].size

    def printSpaceDistr(self):
        grids = self.grids
        x, y, z = [grids[axis] for axis in self.axes]
        MPI.COMM_WORLD.Barrier()
        sleep(0.01*MPI_RANK)
        print(f"Rank {MPI_RANK}({MPI_SIZE}) :\n"
              f"\tx: {x.shape}, [{x.min(initial=np.inf)}, {x.max(initial=-np.inf)}]\n"
              f"\tx: {y.shape}, [{y.min(initial=np.inf)}, {y.max(initial=-np.inf)}]\n"
              f"\tz: {z.shape}, [{z.min(initial=np.inf)}, {z.max(initial=-np.inf)}]\n"
              f"\tcpu: {os.sched_getaffinity(0)}, on {socket.gethostname()}", flush=True)
        MPI.COMM_WORLD.Barrier()


class OutputFiles():
    """
    Utility class used to load and post-process solution
    writen with Dedalus HDF5 IO.
    """
    def __init__(self, folder):
        self.folder = folder
        fileNames = glob.glob(f"{self.folder}/*.h5")
        fileNames.sort(key=lambda f: int(f.split("_s")[-1].split(".h5")[0]))

        assert len(fileNames) == 1, "cannot read fields splitted in several files"
        self.fileName = fileNames[0]

        self._file = None   # temporary buffer to store the HDF5 file

        vData0 = self.file['tasks']['velocity']
        self.x = np.array(vData0.dims[2]["x"])
        self.dim = dim = len(vData0.dims)-2
        if dim == 2:
            self.z = np.array(vData0.dims[3]["z"])
            self.y = self.z
        elif dim == 3:
            self.y = np.array(vData0.dims[3]["y"])
            self.z = np.array(vData0.dims[4]["z"])
        else:
            raise NotImplementedError(f"{dim = }")

    @property
    def file(self):
        if self.file is None:
            self._file = h5py.File(self.fileName, mode='r')
        return self._file

    def __del__(self):
        try:
            self._file.close()
        except:
            pass

    @property
    def nX(self):
        return self.x.size

    @property
    def nY(self):
        return self.y.size

    @property
    def nZ(self):
        return self.z.size

    @property
    def shape(self):
        if self.dim == 2:
            return (4, self.nX, self.nZ)
        elif self.dim == 3:
            return (5, self.nX, self.nY, self.nZ)

    @staticmethod
    def getModes(coord):
        nX = np.size(coord)
        k = np.fft.rfftfreq(nX, 1/nX) + 0.5
        return k

    @property
    def k(self):
        if self.dim == 2:
            return self.getModes(self.x)
        elif self.dim == 3:
            return self.getModes(self.x), self.getModes(self.y)

    @property
    def vData(self):
        return self.file['tasks']['velocity']

    @property
    def bData(self):
        return self.file['tasks']['buoyancy']

    @property
    def pData(self):
        return self.file['tasks']['pressure']

    @property
    def times(self):
        return np.array(self.vData.dims[0]["sim_time"])

    @property
    def nFields(self):
        return len(self.times)

    def readFields(self, iField):
        data = self.file["tasks"]
        fields = [
            data["velocity"][iField, 0],
            data["velocity"][iField, 1],
            ]
        if self.dim == 3:
            fields += [data["velocity"][iField, 2]]
        fields += [
            data["buoyancy"][iField],
            data["pressure"][iField]
            ]
        return np.array(fields)

    def readField(self, name, iBeg=0, iEnd=None, step=1, verbose=False):
        if verbose: print(f"Reading {name} from hdf5 file {self.fileName}")
        if name == "velocity":
            fData = self.vData
        elif name == "buoyancy":
            fData = self.bData
        elif name == "pressure":
            fData = self.pData
        else:
            raise ValueError(f"cannot read {name} from file")
        shape = fData.shape
        if iEnd is None:
            iEnd = shape[0]
        rData = range(iBeg, iEnd, step)
        data = np.empty((len(rData), *shape[1:]), dtype=float)
        for i, iData in enumerate(rData):
            if verbose: print(f" -- field {i+1}/{len(rData)}, idx={iData}")
            data[i] = fData[iData]
        if verbose: print(" -- done !")
        return data


    def getMeanProfiles(self, buoyancy=False, bRMS=False, pressure=False,
                            iBeg=0, iEnd=None, step=1, verbose=False):
        """
        Args:
            iFile (int): file index
            buoyancy (bool, optional): return buoyancy profile. Defaults to False.
            pressure (bool, optional): return pressure profile. Defaults to False.

        Returns:
           profilr (list): mean profiles of velocity, buoyancy and pressure
        """
        profile = []
        axes = 1 if self.dim==2 else (1, 2)
        velocity = self.readField("velocity", iBeg, iEnd, step, verbose)

        # Horizontal mean velocity amplitude
        uH = velocity[:, :self.dim-1]
        meanH = ((uH**2).sum(axis=1)**0.5).mean(axis=axes)
        profile.append(meanH)

        # Vertical mean velocity
        uV = velocity[:, -1]
        meanV = np.mean(abs(uV), axis=axes)
        profile.append(meanV)



        # uRMS = (ux**2 + uz**2).mean(axis=(0, 1))**0.5
        # bRMS = ((b - b.mean(axis=(0, 1)))**2).mean(axis=(0, 1))**0.5

        if bRMS or buoyancy:
            b = self.readField("buoyancy", iBeg, iEnd, step, verbose)
        if buoyancy:
            profile.append(np.mean(b, axis=axes))
        if bRMS:
            diff = b - b.mean(axis=axes)[(slice(None), *[None]*(self.dim-1), slice(None))]
            rms = (diff**2).mean(axis=axes)**0.5
            profile.append(rms)
        if pressure:
            p = self.readField("pressure", iBeg, iEnd, step, verbose)
            profile.append(np.mean(p, axis=axes))          # (time_index, Nz)
        return profile


    def getLayersQuantities(self, iBeg=0, iEnd=None, step=1, verbose=False):
        uMean, _, bRMS = self.getMeanProfiles(
            bRMS=True, iBeg=iBeg, iEnd=iEnd, step=step, verbose=verbose)
        uMean = uMean.mean(axis=0)
        bRMS = bRMS.mean(axis=0)

        z = self.z
        nFine = int(1e4)
        zFine = np.linspace(0, 1, num=nFine)
        P = LagrangeApproximation(z).getInterpolationMatrix(zFine)

        uMeanFine = P @ uMean
        bRMSFine = P @ bRMS

        approx = LagrangeApproximation(z)

        xOptU = sco.minimize_scalar(lambda z: -approx(z, fValues=uMeanFine), bounds=[0, 0.5])
        xOptB = sco.minimize_scalar(lambda z: -approx(z, fValues=bRMS), bounds=[0, 0.5])

        deltaU = xOptU.x
        deltaT = xOptB.x

        return zFine, uMeanFine, bRMSFine, deltaU, deltaT


    @staticmethod
    def decomposeRange(iBeg, iEnd, step, maxSize):
        if iEnd is None:
            raise ValueError("need to provide iEnd for range decomposition")
        nIndices = len(range(iBeg, iEnd, step))
        subRanges = []

        # Iterate over the original range and create sub-ranges
        iStart = iBeg
        while nIndices > 0:
            iStop = iStart + (maxSize - 1) * step
            if step > 0 and iStop > iEnd:
                iStop = iEnd
            elif step < 0 and iStop < iEnd:
                iStop = iEnd

            subRanges.append((iStart, iStop + 1 - (iStop==iEnd), step))
            nIndices -= maxSize
            iStart = iStop + step if nIndices > 0 else iEnd

        return subRanges

    @staticmethod
    def computeMeanSpectrum(uValues, xGrid=None, zGrid=None, verbose=False):
        """ uValues[nT, nVar, nX, (nY,) nZ] """
        uValues = np.asarray(uValues)
        nT, nVar, *gridSizes = uValues.shape
        dim = len(gridSizes)
        assert nVar == dim
        if verbose:
            print(f"Computing Mean Spectrum on u[{', '.join([str(n) for n in uValues.shape])}]")

        energy_spectrum = []
        if dim == 2:

            for i in range(2):
                u = uValues[:, i]                           # (nT, Nx, Nz)
                spectrum = np.fft.rfft(u, axis=-2)          # over Nx -->  #(nT, k, Nz)
                spectrum *= np.conj(spectrum)               # (nT, k, Nz)
                spectrum /= spectrum.shape[-2]              # normalize with Nx --> (nT, k, Nz)
                spectrum = np.mean(spectrum.real, axis=-1)  # mean over Nz --> (nT,k)
                energy_spectrum.append(spectrum)

        elif dim == 3:

            # Check for a cube with uniform dimensions
            nX, nY, nZ = gridSizes
            assert nX == nY
            size = nX // 2

            # Interpolate in z direction
            assert xGrid is not None and zGrid is not None
            if verbose: print(" -- interpolating from zGrid to a uniform mesh ...")

            P = LagrangeApproximation(zGrid, weightComputation="STABLE").getInterpolationMatrix([0.1, 0.5, 0.9])
            uValues = (P @ uValues.reshape(-1, nZ).T).T.reshape(nT, dim, nX, nY, 3)

            # Compute 2D mode disks
            k1D = np.fft.fftfreq(nX, 1/nX)**2
            kMod = k1D[:, None] + k1D[None, :]
            kMod **= 0.5
            idx = kMod.copy()
            idx *= (kMod < size)
            idx -= (kMod >= size)

            idxList = range(int(idx.max()) + 1)
            flatIdx = idx.ravel()

            # Fourier transform and square of Im,Re
            if verbose: print(" -- 2D FFT on u, v & w ...")
            uHat = np.fft.fftn(uValues, axes=(-3, -2))

            if verbose: print(" -- square of Im,Re ...")
            ffts = [uHat[:, i] for i in range(nVar)]
            reParts = [uF.reshape((nT, nX*nY, 3)).real**2 for uF in ffts]
            imParts = [uF.reshape((nT, nX*nY, 3)).imag**2 for uF in ffts]

            # Spectrum computation
            if verbose: print(" -- computing spectrum ...")
            spectrum = np.zeros((nT, size, 3))
            for i in idxList:
                if verbose: print(f" -- k{i+1}/{len(idxList)}")
                kIdx = np.argwhere(flatIdx == i)
                tmp = np.empty((nT, *kIdx.shape, 3))
                for re, im in zip(reParts, imParts):
                    np.copyto(tmp, re[:, kIdx])
                    tmp += im[:, kIdx]
                    spectrum[:, i] += tmp.sum(axis=(1, 2))
            spectrum /= 2*(nX*nY)**2

            energy_spectrum.append(spectrum)
            if verbose: print(" -- done !")

        return energy_spectrum


    def getMeanSpectrum(self, iFile:int, iBeg=0, iEnd=None, step=1, verbose=False, batchSize=5):
        """
        Mean spectrum from a given output file

        Parameters
        ----------
        iFile : int
            Index of the file to use.
        iBeg : int, optional
            Starting index for the fields to use. The default is 0.
        iEnd : int, optional
            Stopping index (non included) for the fields to use. The default is None.
        step : int, optional
            Index step for the fields to use. The default is 1.
        verbose : bool, optional
            Display infos message in stdout. The default is False.
        batchSize : int, optional
            Number of fields to regroup when computing one FFT. The default is 5.

        Returns
        -------
        spectra : np.ndarray[nT,size]
            The spectrum values for all nT fields.
        """
        spectra = []
        if iEnd is None:
            iEnd = self.nFields[iFile]
        subRanges = self.decomposeRange(iBeg, iEnd, step, batchSize)
        for iBegSub, iEndSub, stepSub in subRanges:
            if verbose:
                print(f" -- computing for fields in range ({iBegSub},{iEndSub},{stepSub})")
            velocity = self.readField(iFile, "velocity", iBegSub, iEndSub, stepSub, verbose)
            spectra += computeMeanSpectrum(velocity, verbose=verbose, xGrid=self.x, zGrid=self.z)
        return np.concatenate(spectra)


    def getFullMeanSpectrum(self, iBeg:int, iEnd=None):
        """
        Function to get full mean spectrum

        Args:
            iBeg (int): starting file index
            iEnd (int, optional): stopping file index. Defaults to None.

        Returns:
           sMean (np.ndarray): mean spectrum
           k (np.ndarray): wave number
        """
        if iEnd is None:
            iEnd = self.nFiles
        sMean = []
        for iFile in range(iBeg, iEnd):
            energy_spectrum = self.getMeanSpectrum(iFile)
            sx, sz = energy_spectrum                        # (1,time_index,k)
            sMean.append(np.mean((sx+sz)/2, axis=0))        # mean over time ---> (2, k)
        sMean = np.mean(sMean, axis=0)                      # mean over x and z ---> (k)
        np.savetxt(f'{self.folder}/spectrum.txt', np.vstack((sMean, self.k)))
        return sMean, self.k

    def toVTR(self, idxFormat="{:06d}"):
        """
        Convert all 3D fields from the OutputFiles object into a list
        of VTR files, that can be read later with Paraview or equivalent to
        make videos.

        Parameters
        ----------
        idxFormat : str, optional
            Formating string for the index suffix of the VTR file.
            The default is "{:06d}".

        Example
        -------
        >>> # Suppose the FieldsIO object is already writen into outputs.pysdc
        >>> import os
        >>> from pySDC.utils.fieldsIO import Rectilinear
        >>> os.makedirs("vtrFiles")  # to store all VTR files into a subfolder
        >>> Rectilinear.fromFile("outputs.pysdc").toVTR(
        >>>    baseName="vtrFiles/field", varNames=["u", "v", "w", "T", "p"])
        """
        assert self.dim == 3, "can only be used with 3D fields"
        from pySDC.helpers.vtkIO import writeToVTR

        baseName = f"{self.folder}/vtrFiles"
        os.makedirs(baseName, exist_ok=True)
        baseName += "/out"
        template = f"{baseName}_{idxFormat}"
        coords = [self.x, self.y, self.z]
        varNames = ["velocity_x", "velocity_y", "velocity_z", "buoyancy", "pressure"]
        for i in range(np.cumsum(self.nFields)[0]):
            u = self.fields(i)
            writeToVTR(template.format(i), u, coords, varNames)


if __name__ == "__main__":
    import scipy.optimize as sco

    from pySDC.playgrounds.dedalus.problems.utils import OutputFiles
    # from qmat.lagrange import LagrangeApproximation
    import matplotlib.pyplot as plt

    dirName = "run_3D_A4_R1_M1"

    problem = RBCProblem3D.runSimulation(
        dirName, 100, 1e-2/2, logEvery=20, dtWrite=1.0,
        aspectRatio=4, resFactor=1, meshRatio=1)

    # output = OutputFiles(dirName)
    # approx = LagrangeApproximation(output.z)

    # nThrow = 20
    # nIgnore = 1
    # u = output.vData(0)[nThrow::nIgnore]
    # b = output.bData(0)[nThrow::nIgnore]
    # ux, uz = u[:, 0], u[:, 1]
    # nX, nZ = output.nX, output.nZ

    # # RMS quantities
    # uRMS = (ux**2 + uz**2).mean(axis=(0, 1))**0.5
    # bRMS = ((b - b.mean(axis=(0, 1)))**2).mean(axis=(0, 1))**0.5


    # plt.figure("z-profile")
    # xOptU = sco.minimize_scalar(lambda z: -approx(z, fValues=uRMS), bounds=[0, 0.5])
    # xOptB = sco.minimize_scalar(lambda z: -approx(z, fValues=bRMS), bounds=[0, 0.5])

    # plt.plot(uRMS, output.z, label=f"uRMS[{nX=},{nZ=}]")
    # plt.hlines(xOptU.x, uRMS.min(), uRMS.max(), linestyles="--", colors="black")
    # plt.plot(bRMS, output.z, label=f"bRMS[{nX=},{nZ=}]")
    # plt.hlines(xOptB.x, bRMS.min(), bRMS.max(), linestyles="--", colors="black")
    # plt.legend()


    # keMean = np.sum(mIz*ke, axis=-1).mean(axis=-1)
    # plt.figure("ke")
    # plt.plot(keMean, label=dirName)
    # plt.legend()

    # Removing constant component
    # uAll = uAll - uAll.mean(axis=(-2,-1))[:, None, None]

    # # 2D spectrum
    # mPz = approx.getInterpolationMatrix(np.linspace(0, 1, nZ, endpoint=False))
    # uReg = (mPz @ uAll[:, :, :, None])[..., 0]
    # uHat = np.fft.fft2(uReg)
    # reParts = [uF.ravel().real**2 for uF in uHat]
    # imParts = [uF.ravel().imag**2 for uF in uHat]


    # kX = np.fft.fftfreq(nX, 1/nX)**2
    # kZ = np.fft.fftfreq(nZ, 1/nZ)**2

    # ell = kX[:, None]/(nX//2)**2 + kZ[None, :]/(nZ//2)**2

    # kMod = kX[:, None] + kZ[None, :]
    # kMod **= 0.5

    # idx = kMod.copy()
    # idx *= (ell < 1)
    # idx -= (ell >= 1)
    # idxList = range(int(idx.max()) + 1)
    # flatIdx = idx.ravel()

    # spectrum = np.zeros(len(idxList))
    # for i in idxList:
    #     kIdx = np.argwhere(flatIdx == i)
    #     tmp = np.empty(kIdx.shape)
    #     for re, im in zip(reParts, imParts):
    #         np.copyto(tmp, re[kIdx])
    #         tmp += im[kIdx]
    #         spectrum[i] += tmp.sum()
    # spectrum /= 2*(nX*nZ)**2
    # wavenumbers = list(i + 0.5 for i in idxList)

    # plt.figure("spectrum-2D")
    # plt.loglog(wavenumbers, spectrum)


    # # 1D spectrum (average)
    # spectrum1D = []
    # mIz = approx.getIntegrationMatrix([(0, 1)])
    # uAll = np.concat((u, b[:, None, :, :]), axis=1)
    # for i in range(uAll.shape[1]):
    #     s = np.fft.rfft(uAll[:, i], axis=-2)    # RFFT over Nx --> (nT, Kx, Nz)
    #     s *= np.conj(s)                      # (nT, Kx, Nz)
    #     s = np.sum(mIz*s.real, axis=-1)      # integrate over Nz --> (nT, Kx)
    #     s = s.mean(axis=0)                   # mean over nT -> (Kx,)
    #     s /= nX**2

    #     spectrum1D.append(s)
    # waveNum = np.fft.rfftfreq(nX, 1/nX)+0.5

    # plt.figure("spectrum-1D")
    # plt.loglog(waveNum, spectrum1D[0], label="ux")
    # plt.loglog(waveNum, spectrum1D[1], label="uz")
    # plt.loglog(waveNum, spectrum1D[2], label="b")
    # plt.loglog(waveNum, waveNum**(-5/3), '--k')
    # plt.legend()

    # print(f"iS[ux] : {float(spectrum1D[0].sum())}")
    # print(f"iS[uz] : {float(spectrum1D[1].sum())}")

    # plt.figure(f"contour-{dirName}")
    # plt.pcolormesh(output.x, output.z, b[-1].T)
    # plt.colorbar()
