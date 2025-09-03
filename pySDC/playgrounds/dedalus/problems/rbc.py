#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions building 2D/3D Rayleigh-Benard Convection problems with Dedalus
"""
import os
import socket
from datetime import datetime
from time import sleep

import numpy as np
import dedalus.public as d3
from mpi4py import MPI
import h5py

from pySDC.helpers.fieldsIO import Rectilinear
from pySDC.helpers.blocks import BlockDecomposition
from pySDC.playgrounds.dedalus.timestepper import SDCIMEX, SDCIMEX_MPI, SDCIMEX_MPI2

COMM_WORLD = MPI.COMM_WORLD
MPI_SIZE = COMM_WORLD.Get_size()
MPI_RANK = COMM_WORLD.Get_rank()


class RBCProblem2D():

    def __init__(self, Rayleigh=1e7, Prandtl=1, resFactor=1, meshRatio=4,
                 sComm=COMM_WORLD, mpiBlocks=None, writeSpaceDistr=False,
                 initFields=None, seed=999):

        self.Rayleigh, self.Prandtl = Rayleigh, Prandtl
        self.resFactor = resFactor

        self.infos = {
            "Ra": Rayleigh,
            "Pr": Prandtl,
            }

        self.buildGrid(meshRatio, sComm, mpiBlocks)
        if writeSpaceDistr: self.writeSpaceDistr()
        self.buildProblem()
        self.initFields(initFields, seed)


    def buildGrid(self, meshRatio, sComm, mpiBlocks):
        Lx, Lz = meshRatio, 1
        Nx, Nz = int(64*meshRatio*self.resFactor), int(64*self.resFactor)
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
        p.infos.update(dt=dt, nSteps=nSteps)

        if os.path.isfile(f"{dirName}/01_finalized.txt"):
            if MPI_RANK == 0:
                print(" -- simulation already finalized, skipping !")
            return p
        os.makedirs(dirName, exist_ok=True)
        p.infos.update(dirName=dirName)

        # Solver
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

    def __init__(self, Rayleigh=1e5, Prandtl=0.7, resFactor=1, meshRatio=0.5,
                 sComm=COMM_WORLD, mpiBlocks=None, writeSpaceDistr=False,
                 initFields=None, seed=999):
        super().__init__(
            Rayleigh, Prandtl, resFactor, meshRatio,
            sComm, mpiBlocks, writeSpaceDistr,
            initFields, seed)

    def buildGrid(self, meshRatio, sComm, mpiBlocks):
        Lx, Ly, Lz = 4, 4, 1
        Nx = Ny = int(meshRatio*64*self.resFactor)
        Nz = int(64*self.resFactor)
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


if __name__ == "__main__":
    from pySDC.playgrounds.dedalus.problems.utils import OutputFiles
    from qmat.lagrange import LagrangeApproximation
    import matplotlib.pyplot as plt

    dirName = "test_R4_F2"

    problem = RBCProblem2D.runSimulation(
        dirName, 100, 1e-2/2, logEvery=20, dtWrite=1.0,
        meshRatio=4, resFactor=2)

    output = OutputFiles(dirName)
    approx = LagrangeApproximation(output.z)
    uAll = u, w = output.vData(0)[-1]
    nX, nZ = output.nX, output.nZ

    # Kinetic energy
    mIz = approx.getIntegrationMatrix([(0, 1)])
    ke = (u-u.mean())**2 + (w-w.mean())**2
    keMean = (mIz @ ke.T).mean()
    # 2D spectrum
    uAll = uAll - uAll.mean(axis=(-2,-1))[:, None, None]
    mPz = approx.getInterpolationMatrix(np.linspace(0, 1, nZ, endpoint=False))
    uReg = (mPz @ uAll[:, :, :, None])[..., 0]
    uHat = np.fft.fft2(uReg)
    reParts = [uF.ravel().real**2 for uF in uHat]
    imParts = [uF.ravel().imag**2 for uF in uHat]


    kX = np.fft.fftfreq(nX, 1/nX)**2
    kZ = np.fft.fftfreq(nZ, 1/nZ)**2

    ell = kX[:, None]/(nX//2)**2 + kZ[None, :]/(nZ//2)**2

    kMod = kX[:, None] + kZ[None, :]
    kMod **= 0.5

    idx = kMod.copy()
    idx *= (ell < 1)
    idx -= (ell >= 1)
    idxList = range(int(idx.max()) + 1)
    flatIdx = idx.ravel()

    spectrum = np.zeros(len(idxList))
    for i in idxList:
        kIdx = np.argwhere(flatIdx == i)
        tmp = np.empty(kIdx.shape)
        for re, im in zip(reParts, imParts):
            np.copyto(tmp, re[kIdx])
            tmp += im[kIdx]
            spectrum[i] += tmp.sum()
    spectrum /= 2*(nX*nZ)**2
    wavenumbers = list(i + 0.5 for i in idxList)

    plt.figure("spectrum")
    plt.loglog(wavenumbers, spectrum)

    plt.figure("contour")
    plt.pcolormesh(output.x, output.z, ke.T)
