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

    BASE_RESOLUTION = 64
    """Base number of mesh points in z direction"""


    @staticmethod
    def log(msg):
        if MPI_RANK == 0: print(msg)

    def __init__(self, Rayleigh=1e7, Prandtl=1,
                 resFactor=1, aspectRatio=4, meshRatio=1,
                 sComm=COMM_WORLD, mpiBlocks=None, printSpaceDistr=False,
                 initField=None, seed=999):

        self.Rayleigh, self.Prandtl = Rayleigh, Prandtl
        self.resFactor = resFactor

        self.infos = {
            "Ra": Rayleigh,
            "Pr": Prandtl,
            }

        self.log(" -- building grid ...")
        self.buildGrid(aspectRatio, meshRatio, sComm, mpiBlocks)
        self.log(f" -- {self.dim}D grid done !")

        if printSpaceDistr: self.printSpaceDistr()

        self.log(" -- building problem ...")
        self.buildProblem()
        self.log(" -- done !")

        self.initField(initField, seed)


    def buildGrid(self, aspectRatio, meshRatio, sComm, mpiBlocks):
        baseSize = self.BASE_RESOLUTION
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


    def initField(self, initPath, seed):
        b, u = self.fields["buoyancy"], self.fields["velocity"]
        z, Lz = self.z, self.Lz

        if initPath is None:  # linear buoyancy with random noise
            self.log(" -- generating randomly perturbed initial field ...")
            b.fill_random('g', seed=seed, distribution='normal', scale=1e-3) # Random noise
            b['g'] *= z * (Lz - z) # Damp noise at walls
            b['g'] += Lz - z # Add linear background
        else:
            if os.path.isdir(initPath): # use OutputFiles format
                initPath = OutputFiles(initPath)
                self.log(" -- reading field from HDF5 file ...")
                uInit = initPath.file['tasks']
                for name, field in self.fields.items():
                    localSlices = (slice(None),) * len(field.tensorsig) \
                        + self.dist.grid_layout.slices(field.domain, field.scales)
                    try:
                        field['g'] = uInit[name][(-1, *localSlices)]
                    except KeyError:
                        self.log(f" -- {name} not present in file, putting zeros instead")
                        field['g'] = 0
            elif initPath.lower().endswith(".pysdc"):
                self.log(" -- reading field from pySDC file ...")
                initPath = Rectilinear.fromFile(initPath)
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
                _, uInit = initPath.readField(-1)
                for name, field in self.fields.items():
                    try:
                        field['g'] = uInit[sFields[name]]
                    except KeyError:
                        self.log(f" -- {name} not present in file, putting zeros instead")
                        field['g'] = 0
            else:
                raise ValueError(f"unknown type for initField ({initPath})")
        self.log(" -- done !")
        self.fields0 = {name: field.copy() for name, field in self.fields.items()}

    @classmethod
    def runSimulation(cls, runDir, tEnd, baseDt, tBeg=0, logEvery=100,
                      dtWrite=None, writeVort=False, writeTau=False,
                      timeScheme="RK443", timeParallel=False, groupTimeProcs=False,
                      **pParams):

        cls.log(f"RBC simulation in {runDir}")

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
        cls.log(f" -- selected time-stepper : {timeStepper}")

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
            cls.log(f" -- activated PinT for SDC : {timeParallel}")

        p = cls(**pParams)

        dt = baseDt/p.resFactor
        nSteps = round(float(tEnd-tBeg)/dt, ndigits=3)
        if float(tEnd-tBeg) != round(nSteps*dt, ndigits=3):
            raise ValueError(f"{tEnd=} is not divisible by timestep {dt=} ({nSteps=})")
        nSteps = int(nSteps)
        p.infos.update(tEnd=tEnd, dt=dt, nSteps=nSteps)

        if os.path.isfile(f"{runDir}/01_finalized.txt"):
            cls.log(" -- simulation already finalized, skipping !")
            return p
        os.makedirs(runDir, exist_ok=True)
        p.infos.update(dirName=runDir)

        # Solver
        cls.log(" -- building dedalus solver ...")
        solver = p.problem.build_solver(timeStepper)
        solver.sim_time = tBeg
        solver.stop_sim_time = tEnd
        cls.log(" -- done !")

        # Fields IO
        if dtWrite:
            cls.log(" -- setting up fields output ...")
            iterWrite = dtWrite/dt
            if int(iterWrite) != round(iterWrite, ndigits=3):
                raise ValueError(f"{dtWrite=} is not divisible by {dt=} ({iterWrite=})")
            iterWrite = int(iterWrite)
            snapshots = solver.evaluator.add_file_handler(
                runDir, sim_dt=dtWrite, max_writes=tEnd/dt)
            for name in ["velocity", "buoyancy", "pressure"]:
                snapshots.add_task(p.fields[name], name=name)
            if writeTau:
                for name in ["tau_p", "tau_b1", "tau_b2", "tau_u1", "tau_u2"]:
                    snapshots.add_task(p.fields[name], name=name)
            if writeVort:
                u = p.fields["velocity"]
                snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
            cls.log(" -- done !")


        if MPI_RANK == 0:
            with open(f"{runDir}/00_infoSimu.txt", "w") as f:
                for key, val in p.infos.items():
                    if type(val) == float:
                        f.write(f"{key} : {val:1.2e}\n")
                    else:
                        f.write(f"{key} : {val}\n")

        def log(msg):
            if MPI_RANK == 0:
                with open(f"{runDir}/simu.log", "a") as f:
                    f.write(f"{runDir} -- ")
                    f.write(datetime.now().strftime("%d/%m/%Y  %H:%M:%S"))
                    f.write(f", MPI rank {MPI_RANK} ({MPI_SIZE})")
                    f.write(f" : {msg}\n")

        # Main loop
        if nSteps == 0:
            return p
        try:
            cls.log(f" -- starting simulation (see {runDir}/simu.log) ...")
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
                with open(f"{runDir}/01_finalized.txt", "w") as f:
                    f.write("Done !")
            cls.log(" -- done !")
        except:
            log('Exception raised, triggering end of main loop.')
            if MPI_SIZE > 1:
                COMM_WORLD.Abort()
            raise
        finally:
            solver.log_stats()

        return p


class RBCProblem3D(RBCProblem2D):

    BASE_RESOLUTION = 32
    """Base number of mesh points in z direction"""

    def __init__(self, Rayleigh=1e5, Prandtl=0.7,
                 resFactor=1, aspectRatio=4, meshRatio=0.5,
                 sComm=COMM_WORLD, mpiBlocks=None, printSpaceDistr=False,
                 initField=None, seed=999):
        super().__init__(
            Rayleigh, Prandtl,
            resFactor, aspectRatio, meshRatio,
            sComm, mpiBlocks, printSpaceDistr,
            initField, seed)

    def buildGrid(self, aspectRatio, meshRatio, sComm, mpiBlocks):
        baseSize = self.BASE_RESOLUTION

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
        self.log(f" -- {mpiBlocks = }")

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

        with open(f"{self.folder}/00_infoSimu.txt", "r") as f:
            lines = f.readlines()
        self.Ra = float(lines[0].split(" : ")[-1])
        self.Pr = float(lines[1].split(" : ")[-1])

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
    def kappa(self) -> np.ndarray:
        return np.arange(self.nX//2) + 0.5

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


        if which == "all":
            which = ["ke", "NuV", "NuT", "NuB"]
        else:
            which = list(which)

        series = {name: [] for name in which}
        avgAxes = 1 if self.dim==2 else (1, 2)

        approx = LagrangeApproximation(self.z)
        mIz = approx.getIntegrationMatrix([(0, 1)])
        if set(which).intersection(["NuV", "NuT", "NuB"]):
            mDz = approx.getDerivativeMatrix()
            if "NuB" in which:
                mDzB = approx.getInterpolationMatrix([0]) @ mDz
            if "NuT" in which:
                mDzT = approx.getInterpolationMatrix([1]) @ mDz

        for r in self.BatchRanges(self.nFields, maxSize=batchSize):

            u = self.readFields("velocity", r.start, r.stop, r.step)
            w = u[:, -1]

            if set(which).intersection(["NuV", "NuT", "NuB"]):
                b = self.readFields("buoyancy", r.start, r.stop, r.step)

            if "NuV" in which:
                coeff = (self.Ra*self.Pr)**0.5
                integ = w*b*coeff - (mDz @ b[..., None])[..., 0]
                nuV = mIz @ integ.mean(axis=avgAxes)[..., None]
                series["NuV"].append(nuV.ravel())

            if "NuT" in which:
                nuT = - mDzT @ b.mean(axis=avgAxes)[..., None]
                series["NuT"].append(nuT.ravel())

            if "NuB" in which:
                nuB = - mDzB @ b.mean(axis=avgAxes)[..., None]
                series["NuB"].append(nuB.ravel())

            if "ke" in which:
                u **= 2
                ke = mIz @ u.sum(axis=1).mean(axis=avgAxes)[..., None]
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


    def getBoundaryLayers(self, which=["uRMS", "bRMS"], profiles=None,
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

    def getSpectrum(self, which=["uv", "uh"], zVal="all",
                    start=0, stop=None, step=1, batchSize=None):
        if stop is None:
            stop = self.nFields
        if which == "all":
            which = ["uv", "uh", "b"]
        else:
            which = list(which)

        kappa = self.kappa
        spectrum = {name: np.zeros(kappa.size) for name in which}

        approx = LagrangeApproximation(self.z, weightComputation="STABLE")
        if zVal == "all":
            mIz = approx.getIntegrationMatrix([(0, 1)])
        else:
            mPz = approx.getInterpolationMatrix([zVal, 1-zVal])

        nSamples = 0
        for r in self.BatchRanges(start, stop, step, batchSize):

            bSize = len(r)

            for name in which:

                # read fields
                if name in ["uv", "uh"]:
                    u = self.readFields("velocity", r.start, r.stop, r.step)
                    if name == "uv":
                        field = u[:, -1:]
                    if name == "uh":
                        field = u[:, :-1]
                if name == "b":
                    field = self.readFields("buoyancy", r.start, r.stop, r.step)[:, None, ...]
                    if self.dim == 3:
                        field -= field.mean(axis=(2, 3))[:, :, None, None, :]
                    else:
                        field -= field.mean(axis=2)[:, :, None, :]
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

                    # ignore last frequency (zero amplitude)
                    s = s[:, :-1]

                # 3D case
                elif self.dim == 3:

                    if self.VERBOSE:
                        print(f" -- computing 2D mean spectrum for {name}[{start},{stop},{step}] ...")

                    assert self.nX == self.nY, "nX != nY, that will be some weird spectrum"
                    nT, nVar, nX, nY, nZ = field.shape

                    size = kappa.size

                    # compute 2D mode disks
                    k1D = np.fft.fftfreq(nX, 1/nX)**2
                    kMod = k1D[:, None] + k1D[None, :]
                    kMod **= 0.5

                    idx = kMod.copy()
                    np.trunc(idx, out=idx)
                    idx *= (kMod < size)
                    idx -= (kMod >= size)

                    idxList = range(int(idx.max()) + 1)
                    flatIdx = idx.ravel()

                    if self.VERBOSE: print(" -- 2D FFT ...")
                    uHat = np.fft.fft2(field, axes=(-3, -2))

                    if self.VERBOSE: print(" -- square of Im,Re ...")
                    ffts = [uHat[:, i] for i in range(nVar)]
                    reParts = [uF.reshape((nT, nX*nY, -1)).real**2 for uF in ffts]
                    imParts = [uF.reshape((nT, nX*nY, -1)).imag**2 for uF in ffts]

                    # spectrum computation
                    if self.VERBOSE: print(" -- computing spectrum ...")
                    s = np.zeros((nT, size, nZ))
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

                # add batch mean over nT
                sMean = spectrum[name]
                sMean *= nSamples
                s = s.mean(axis=0)
                s *= bSize
                sMean += s
                sMean /= (nSamples + bSize)

                if self.VERBOSE: print(" -- done !")

            nSamples += bSize

        return spectrum


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

def checkDNS(spectrum:np.ndarray, kappa:np.ndarray, sRatio:int=4, nThrow:int=0):
    r"""
    Check for a well-resolved DNS, by looking at an energy spectrum
    :math:`s(\kappa)` and doing a quadratic regression in log space
    on its tail :

    .. math::
        \log(s_{tail}) \simeq
        a\log(\kappa_{tail})^2 + b\log(\kappa_{tail}) + c

    where :math:`(\kappa_{tail},s_{tail})` is continuous subset of the
    mapping :math:`(\kappa,s)` for large values of :math:`\kappa`
    (i.e spectrum tail).
    If the quadratic regression produces a convex polynomial
    (i.e :math:`a > 0`) then the simulation is considered as under-resolved
    (no DNS).
    Per default, the tail is built considering the
    **last quarter of the spectrum**.

    Parameters
    ----------
    spectrum : np.ndarray
        Vector of the spectrum values :math:`s(\kappa)`.
    kappa : np.ndarray
        Vector of the wavenumber values :math:`\kappa`.
    sRatio : int, optional
        Spectrum ratio used to define the tail: if the spectrum has
        N values, then the tail is defined with the last N/sRatio values.
        The default is 4.
    nThrow : int, optional
        Number of higher kappa extremity spectrum values to not consider
        when defining the tail. The default is 1.

    Returns
    -------
    results : dict
        Dictionnary containing the results, with keys :

        - `DNS` : boolean indicating if the simulation is well resolved
        - `coeffs` : the :math:`a,b,c` regression coefficients, stored in a tuple
        - `kTail` : the :math:`\kappa_{tail}` values used for the regression
        - `sTail` : the :math:`s_{tail}` values used for the regression

    """
    spectrum = np.asarray(spectrum)
    kappa = np.asarray(kappa)
    assert spectrum.ndim == 1, "spectrum must be a 1D vector"
    assert kappa.shape == spectrum.shape, "kappa and spectrum must have the same shape"

    nValues = kappa.size//sRatio
    sl = slice(-nValues-nThrow, -nThrow if nThrow else None)

    kTail = kappa[sl]
    sTail = spectrum[sl]

    y = np.log(sTail)
    x = np.log(kTail)

    def fun(coeffs):
        a, b, c = coeffs
        return np.linalg.norm(y - a*x**2 - b*x - c)

    res = sco.minimize(fun, [0, 0, 0])
    a, b, c = res.x

    results = {
        "DNS": not a > 0,
        "coeffs": (a, b, c),
        "kTail": kTail,
        "sTail": sTail,
    }

    return results

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # dirName = "run_3D_A4_M0.5_R1_Ra1e6"
    dirName = "run_3D_A4_M1_R1_Ra1e6"
    # dirName = "run_M4_R2"
    # dirName = "test_M4_R2"
    OutputFiles.VERBOSE = True
    output = OutputFiles(dirName)

    if False:
        series = output.getTimeSeries(which=["NuV", "NuT", "NuB"])

        plt.figure("series")
        for name, values in series.items():
            plt.plot(output.times, values, label=name)
        plt.legend()

    start = 60

    if False:
        which = ["bRMS"]


        Nu = series["NuV"][start:].mean()

        profiles = output.getProfiles(
            which, start=start, batchSize=None)
        deltas = output.getBoundaryLayers(
            which, start=start, profiles=profiles)

        for name, p in profiles.items():
            if "Mean" in name:
                plt.figure("Mean profiles")
                plt.plot(p, output.z, label=name)
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

        zLog = np.logspace(np.log10(1/(100*Nu)), np.log(0.5), num=200)
        approx = LagrangeApproximation(output.z)
        mPz = approx.getInterpolationMatrix(zLog)

        bMean = (profiles["bMean"] + (1-profiles["bMean"][-1::-1]))/2
        bMean = mPz @ bMean

        bRMS = (profiles["bRMS"] + profiles["bRMS"][-1::-1])/2
        bRMS = mPz @ bRMS

        plt.figure("mean-log")
        plt.semilogx(zLog*Nu, bMean, label=dirName)
        plt.legend()

        plt.figure("rms-log")
        plt.semilogx(zLog*Nu, bRMS, label=dirName)
        plt.legend()

    if True:
        spectrum = output.getSpectrum(
            which=["uh"], zVal="all", start=start, batchSize=None)
        kappa = output.kappa

        check = checkDNS(spectrum["uh"], kappa)
        print(f"DNS : {check['DNS']}")
        a, b, c = check["coeffs"]
        kTail = check["kTail"]
        sTail = check["sTail"]

        plt.figure("spectrum")
        for name, vals in spectrum.items():
            plt.loglog(kappa[1:], vals[1:], label=name)
        plt.loglog(kTail, sTail, '.', c="black")
        kTL = np.log(kTail)
        plt.loglog(kTail, np.exp(a*kTL**2 + b*kTL + c), c="gray")
        plt.loglog(kappa[1:], kappa[1:]**(-5/3), '--k')
        plt.legend()
