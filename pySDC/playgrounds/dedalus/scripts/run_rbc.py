#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json


parser = argparse.ArgumentParser(
    description='Run RBC simulation using Dedalus',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "runDir", help="name of the simulation directory")
parser.add_argument(
    "--baseDt", "-dt", help="time-step used with resFactor=1",
    default=1e-2/2, type=float)
parser.add_argument(
    "--tEnd", "-t", help="end simulation time (tBeg=0)",
    default=100, type=float)

parser.add_argument(
    "--dim", "-d", help="dimension for the simulation",
    default=2, type=int, choices=[2, 3])
parser.add_argument(
    "--aspectRatio", "-ar", help="geometric aspect ratio A, Lx = Ly = A*Lz",
    default=4, type=float)
parser.add_argument(
    "--meshRatio", "-mr", help="mesh point ratio M, Nx/Lx = Ny/Ly = M*Nz/Lz",
    default=1, type=float)
parser.add_argument(
    "--resFactor", "-rf", help="resolution factor R, Nz = 64*R (2D) or 32*R (3D)",
    default=1, type=float)
parser.add_argument(
    "--Rayleigh", "-Ra", help="Rayleigh number Ra",
    default=1e7, type=float)
parser.add_argument(
    "--Prandtl", "-Pr", help="Prandtl number Pr",
    default=1, type=float)
parser.add_argument(
    "--initField", "-if", help="path for the initial field",
    default=None)

parser.add_argument(
    "--dtWrite", "-dtw", help="time-step for writing solution output",
    type=float)
parser.add_argument(
    "--logEvery", "-l", help="log every [...] time-steps",
    default=100, type=int)

parser.add_argument(
    "--timeScheme", help="time integration method to be used",
    choices=["RK111", "RK222", "RK443", "SDC"])
parser.add_argument(
    "--timeParallel", help="which time-parallelization to use with SDC",
    choices=["MPI", "MPI2"], default=False)
parser.add_argument(
    "--groupTimeProcs", help="wether or not grouping the time processes",
    action="store_true")
parser.add_argument(
    "--writeDecomposition", help="write the parallel space-time decomposition in a file",
    action="store_true")

parser.add_argument(
    "--nNodesSDC", help="number of time nodes per step for SDC",
    default=4, type=int)
parser.add_argument(
    "--nSweepsSDC", help="number of sweeps per time step for SDC",
    default=4, type=int)
parser.add_argument(
    "--implSweepSDC", help="implicit sweep type for SDC",
    default="MIN-SR-S")
parser.add_argument(
    "--explSweepSDC", help="explicit sweep type for SDC",
    default="PIC")

args = parser.parse_args()

# imports library after parsing args (do not import with --help)
from pySDC.playgrounds.dedalus.timestepper import SDCIMEX, MPI
from pySDC.playgrounds.dedalus.problems.rbc import RBCProblem2D, RBCProblem3D

params = args.__dict__

dim = params.pop("dim")
ProblemClass = RBCProblem2D if dim == 2 else RBCProblem3D
SDCIMEX.setParameters(
    nNodes=params.pop("nNodesSDC"),
    nodeType="LEGENDRE", quadType="RADAU-RIGHT",
    nSweeps=params.pop("nSweepsSDC"),
    initSweep="COPY",
    implSweep=params.pop("implSweepSDC"), explSweep=params.pop("explSweepSDC"),
)


prob = ProblemClass.runSimulation(**params)
if MPI.COMM_WORLD.Get_rank() == 0:
    with open(f"{args.runDir}/infos.json", "w") as f:
        json.dump(prob.infos, f)
