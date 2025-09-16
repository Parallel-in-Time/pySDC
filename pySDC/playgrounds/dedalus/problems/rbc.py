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
        unitSize = baseSize*self.resFactor

        Nx = Ny = int(aspectRatio*meshRatio*unitSize)
        Nz = int(unitSize)
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
    VERBOSE = False

    def __init__(self, folder):
        self.folder = folder
        fileNames = glob.glob(f"{self.folder}/*.h5")
        fileNames.sort(key=lambda f: int(f.split("_s")[-1].split(".h5")[0]))

        assert len(fileNames) == 1, "cannot read fields splitted in several files"
        self.fileName = fileNames[0]

        self.file = h5py.File(self.fileName, mode='r')

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

    def __del__(self):
        try:
            self.file.close()
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

    @property
    def waveNumbers(self) -> np.ndarray:
        nX = self.nX
        k = np.fft.rfftfreq(nX, 1/nX) + 0.5
        return k

    @property
    def vData(self):
        return self.file['tasks']['velocity']

    @property
    def bData(self):
        return self.file['tasks']['buoyancy']

    @property
    def pData(self):
        return self.file['tasks']['pressure']

    def readFieldAt(self, iField):
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

    @property
    def times(self):
        return np.array(self.vData.dims[0]["sim_time"])

    @property
    def nFields(self):
        return len(self.times)


    class BatchRanges():

        def __init__(self, *args, maxSize=None, start=0, step=1):
            assert len(args) < 5
            try:
                start, stop, step, maxSize = args
            except:
                try:
                    start, stop, step = args
                except:
                    try:
                        start, stop = args
                    except:
                        stop, = args
            self.range = range(start, stop, step)
            self.maxSize = stop if not maxSize else maxSize

        @property
        def start(self):
            return self.range.start

        @property
        def stop(self):
            return self.range.stop

        @property
        def step(self):
            return self.range.step

        def __len__(self):
            return len(self.range)

        def __iter__(self):
            for i in range(self.start, self.stop, self.step*self.maxSize):
                yield range(i, min(i+self.maxSize*self.step, self.stop), self.step)


    def readFields(self, name, start=0, stop=None, step=1):
        if self.VERBOSE: print(f"Reading {name} from hdf5 file {self.fileName}")
        if name == "velocity":
            field = self.vData
        elif name == "buoyancy":
            field = self.bData
        elif name == "pressure":
            field = self.pData
        else:
            raise ValueError(f"cannot read {name} from file")
        if stop is None:
            stop = self.nFields
        if self.VERBOSE: print(f" -- reading {name}[{start}:{stop}:{step}] ...")
        data = field[start:stop:step]
        if self.VERBOSE: print(" -- done !")
        return data


    def getTimeSeries(self, which=["ke"], batchSize=None):

        series = {name: [] for name in which}
        avgAxes = 1 if self.dim==2 else (1, 2)

        Iz = LagrangeApproximation(self.z).getIntegrationMatrix([(0, 1)])

        for r in self.BatchRanges(self.nFields, maxSize=batchSize):

            if "ke" in which:
                u = self.readFields("velocity", r.start, r.stop, r.step)
                u **= 2
                ke = Iz @ u.sum(axis=1).mean(axis=avgAxes)[..., None]
                series["ke"].append(ke.ravel())

        for key, val in series.items():
            series[key] = np.array(val).ravel()

        return series


    def getProfiles(self, which=["uRMS", "bRMS"],
                    start=0, stop=None, step=1, batchSize=None):
        if stop is None:
            stop = self.nFields
        avgAxes = (0, 1) if self.dim==2 else (0, 1, 2)
        formula = {
            "u" : "u.sum(axis=1)",
            "uv": "u[:, -1]",
            "uh": "u[:, :-1].sum(axis=1)",
            "b": "b",
            "p": "p",
            }

        if which == "all":
            which = [var+"Mean" for var in formula.keys()] \
                + [var+"RMS" for var in formula.keys()]
        else:
            which = list(which)
        if "bRMS" in which and "bMean" not in which:
            which.append("bMean")
        if "pRMS" in  which and "pMean" not in which:
            which.append("pMean")
        profiles = {name: np.zeros(self.nZ) for name in which}

        nSamples = 0
        def addSamples(current, new, nNew):
            current *= nSamples
            new *= nNew
            current += new
            current /= (nSamples + nNew)

        # Mean profiles
        for r in self.BatchRanges(start, stop, step, batchSize):

            bSize = len(r)

            # Read required data
            if set(["uMean", "uvMean", "uhMean",
                    "uRMS", "uvRMS", "uhRMS"]).intersection(which):
                u = self.readFields("velocity", r.start, r.stop, r.step)
            if set(["bMean"]).intersection(which):
                b = self.readFields("buoyancy", r.start, r.stop, r.step)
            if set(["pMean"]).intersection(which):
                p = self.readFields("pressure", r.start, r.stop, r.step)

            # Mean profiles
            for name in which:
                if "Mean" in name:
                    var = eval(formula[name[:-4]]).mean(axis=avgAxes)
                    addSamples(profiles[name], var, bSize)

            try: u **= 2
            except: pass

            # RMS profiles
            for name in which:
                if "RMS" in name and name not in ["bRMS", "pRMS"]:
                    var = eval(formula[name[:-3]]).mean(axis=avgAxes)
                    addSamples(profiles[name], var, bSize)

            nSamples += bSize

        # bRMS and pRMS require precomputed mean
        nSamples = 0
        for r in self.BatchRanges(start, stop, step, batchSize):

            bSize = len(r)

            if "bRMS" in which:
                b = self.readFields("buoyancy", r.start, r.stop, r.step)
                b -= profiles["bMean"]
                b **= 2
                bRMS = b.mean(axis=avgAxes)
                addSamples(profiles["bRMS"], bRMS, bSize)

            if "pRMS" in which:
                p = self.readFields("pressure", r.start, r.stop, r.step)
                p -= profiles["pMean"]
                p **= 2
                pRMS = p.mean(axis=avgAxes)
                addSamples(profiles["pRMS"], pRMS, bSize)

            nSamples += bSize

        # Square root after time averaging for RMS
        for name, val in profiles.items():
            if "RMS" in name:
                val **= 0.5

        profiles["nSamples"] = nSamples
        return profiles


    def getBoundaryLayers(self,
                          which=["uRMS", "bRMS"], profiles=None,
                          start=0, stop=None, step=1, batchSize=None):

        if which == "all":
            which = ["uRMS", "bRMS", "uhRMS"]
        else:
            which = list(which)
        deltas = {name: None for name in which}

        if profiles is None:
            profiles = {}
        missing = set(which).difference(profiles.keys())
        profiles.update(self.getProfiles(missing, start, stop, step, batchSize))

        approx = LagrangeApproximation(self.z)

        for name in which:
            values = profiles[name]
            opt = sco.minimize_scalar(lambda z: -approx(z, fValues=values), bounds=[0, 0.5])
            deltas[name] = opt.x

        return deltas

    def computeSpectrum(self, which=["uv", "uh"], zVal="all",
                        start=0, stop=None, step=1, batchSize=None):

        if which == "all":
            which = ["uv", "uh", "b"]
        else:
            which = list(which)
        waveNum = self.waveNumbers
        spectrum = {name: np.zeros(waveNum.size) for name in which}

        approx = LagrangeApproximation(self.z, weightComputation="STABLE")
        if zVal == "all":
            mIz = approx.getIntegrationMatrix([(0, 1)])
        else:
            mPz = approx.getInterpolationMatrix([zVal, 1-zVal])

        for name in which:

            # read fields
            if name in ["uv", "uh"]:
                u = self.readFields("velocity", start, stop, step)
                if name == "uv":
                    field = u[:, -1:]
                if name == "uh":
                    field = u[:, :-1]
            if name == "b":
                field = self.readFields("buoyancy", start, stop, step)[:, None, ...]
            # field.shape = (nT,nVar,nX[,nY],nZ)

            if zVal != "all":
                field = (mPz @ field[..., None])[..., 0]

            # 2D case
            if self.dim == 2:
                if self.VERBOSE:
                    print(f" -- computing 1D mean spectrum for {name}[{start},{stop},{step}] ...")
                var = field[:, 0]

                # RFFT over nX --> (nT, nKx, nZ)
                s = np.fft.rfft(var, axis=-2)
                s *= np.conj(s)

                # scaling
                s /= self.nX**2

            # 3D case
            elif self.dim == 3:

                if self.VERBOSE:
                    print(f" -- computing 2D mean spectrum for {name}[{start},{stop},{step}] ...")

                assert self.nX == self.nY, "nX != nY, that will be some weird spectrum"
                nT, nVar, nX, nY, nZ = field.shape

                # Compute 2D mode disks
                k1D = np.fft.fftfreq(nX, 1/nX)**2
                kMod = k1D[:, None] + k1D[None, :]
                kMod **= 0.5
                idx = kMod.copy()
                idx *= (kMod < waveNum.size)
                idx -= (kMod >= waveNum.size)

                idxList = range(int(idx.max()) + 1)
                flatIdx = idx.ravel()

                if self.VERBOSE: print(" -- 2D FFT ...")
                uHat = np.fft.fft2(field, axes=(-3, -2))

                if self.VERBOSE: print(" -- square of Im,Re ...")
                ffts = [uHat[:, i] for i in range(nVar)]
                reParts = [uF.reshape((nT, nX*nY, -1)).real**2 for uF in ffts]
                imParts = [uF.reshape((nT, nX*nY, -1)).imag**2 for uF in ffts]

                # Spectrum computation
                if self.VERBOSE: print(" -- computing spectrum ...")
                s = np.zeros((nT, waveNum.size, nZ))
                for i in idxList:
                    if self.VERBOSE: print(f" -- k{i+1}/{len(idxList)}")
                    kIdx = np.argwhere(flatIdx == i)
                    tmp = np.empty((nT, *kIdx.shape, nZ))
                    for re, im in zip(reParts, imParts):
                        np.copyto(tmp, re[:, kIdx])
                        tmp += im[:, kIdx]
                        s[:, i] += tmp.sum(axis=(1, 2))

                # scaling
                s /= 2*(nX*nY)**2

            # integral or mean over nZ --> (nT, Kx)
            if zVal == "all":
                s = (mIz @ s.real[..., None])[..., 0, 0]
            else:
                s = s.real.mean(axis=-1)

            # mean over nT
            spectrum[name] += s.mean(axis=0)

            if self.VERBOSE: print(" -- done !")

        return spectrum


    def getMeanSpectrum(self, iFile:int, iBeg=0, iEnd=None, step=1, batchSize=5, verbose=False):
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
            spectra += self.computeMeanSpectrum(velocity, verbose=verbose, xGrid=self.x, zGrid=self.z)
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
        for i in range(self.nFields):
            u = self.readFieldAt(i)
            writeToVTR(template.format(i), u, coords, varNames)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    aspectRatio = 4     # Lx = Ly = A*Lz
    meshRatio = 0.5     # Nx/Lx = Ny/Ly = M*Nz/Lz
    resFactor = 1       # Nz = R*32

    dirName = f"run_3D_A{aspectRatio}_M{meshRatio}_R{resFactor}"

    # problem = RBCProblem3D.runSimulation(
    #     dirName, 100, 1e-2/4, logEvery=20, dtWrite=1.0,
    #     aspectRatio=4, meshRatio=1, resFactor=1)


    dirName = "run_3D_A4_M0.5_R1"
    # dirName = "run_M4_R2"
    OutputFiles.VERBOSE = True
    output = OutputFiles(dirName)

    if False:
        series = output.getTimeSeries()

        plt.figure("series")
        plt.plot(output.times, series["ke"], label="ke")
        plt.legend()

        profiles = output.getProfiles(
            which="all", batchSize=None, start=30, step=5)

        deltas = output.getBoundaryLayers(which="all", profiles=profiles)

        for name, p in profiles.items():
            if "Mean" in name:
                plt.figure("Mean profiles")
                plt.plot(p, output.z, label=name)
                if name in deltas:
                    plt.hlines(deltas[name], p.min(), p.max(), linestyles="--", colors="black")
            if "RMS" in name:
                plt.figure("RMS profiles")
                plt.plot(p, output.z, label=name)
                if name in deltas:
                    plt.hlines(deltas[name], p.min(), p.max(), linestyles="--", colors="black")

        for pType in ["Mean", "RMS"]:
            plt.figure(f"{pType} profiles")
            plt.legend()
            plt.xlabel("profile")
            plt.ylabel("z coord")

    spectrum = output.computeSpectrum(which="all", zVal="all")
    waveNum = output.waveNumbers

    plt.figure("spectrum")
    for name, vals in spectrum.items():
        plt.loglog(waveNum, vals, label=name)
    plt.loglog(waveNum, waveNum**(-5/3), '--k')
    plt.legend()


    if False:
        approx = LagrangeApproximation(output.z)

        start = 20
        stop = 1
        u = output.vData[start::stop]
        b = output.bData[start::stop]
        ux, uz = u[:, 0], u[:, 1]
        nX, nZ = output.nX, output.nZ

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
        spectrum1D = []
        mIz = approx.getIntegrationMatrix([(0, 1)])
        uAll = np.concat((u, b[:, None, ...]), axis=1)
        for i in range(uAll.shape[1]):
            s = np.fft.rfft(uAll[:, i], axis=-2)      # RFFT over Nx --> (nT, Kx, Nz)
            s *= np.conj(s)                           # (nT, Kx, Nz)
            s = (mIz @ s[..., None])[..., 0, 0]       # integrate over Nz --> (nT, Kx)
            s /= nX**2
            s = s.mean(axis=0)                        # mean over nT -> (Kx,)


            spectrum1D.append(s)
        waveNum = np.fft.rfftfreq(nX, 1/nX)+0.5

        # print(f"iS[ux] : {float(spectrum1D[0].sum())}")
        # print(f"iS[uz] : {float(spectrum1D[1].sum())}")

        # plt.figure(f"contour-{dirName}")
        # plt.pcolormesh(output.x, output.z, b[-1].T)
        # plt.colorbar()
