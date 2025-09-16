#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

from pySDC.helpers.fieldsIO import FieldsIO
from pySDC.playgrounds.dedalus.problems.rbc import RBCProblem2D, RBCProblem3D, OutputFiles

parser = argparse.ArgumentParser(
    description='Run RBC simulation using Dedalus',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "runDir", help="name of the simulation directory")

parser.add_argument(
    "--dim", "-d", help="dimension for the simulation",
    default=2, type=int, choices=[2, 3])
parser.add_argument(
    "--aspectRatio", "-ar", help="geometric aspect ratio, Lx = Ly = ar*Lz",
    default=4, type=float)
parser.add_argument(
    "--meshRatio", "-mr", help="mesh point ratio, Nx/Lx = Ny/Ly = mr*Nz/Lz",
    default=1, type=float)
parser.add_argument(
    "--resFactor", "-rf", help="resolution factor, Nz = 64*rf (2D) or 32*rf (3D)",
    default=1, type=float)
parser.add_argument(
    "--Rayleigh", "-ra", help="Rayleigh number",
    default=1e7, type=float)
parser.add_argument(
    "--Prandtl", "-pr", help="Prandtl number",
    default=1, type=float)
parser.add_argument(
    "--initField", "-if", help="path for the initial field",
    default=None)

parser.add_argument(
    "--baseDt", "-dt", help="time-step used with resFactor=1",
    default=1e-2/2, type=float)
parser.add_argument(
    "--tEnd", "-t", help="end simulation time (tBeg=0)",
    default=100, type=float)
parser.add_argument(
    "--dtWrite", "-dtw", help="time-step for writing solution output",
    default=1, type=float)
parser.add_argument(
    "--logEvery", "-l", help="log every [...] time-steps",
    default=100, type=int)

args = parser.parse_args(["test"])
params = args.__dict__

dim = params.pop("dim")
ProblemClass = RBCProblem2D if dim == 2 else RBCProblem3D


ProblemClass.runSimulation(**params)
