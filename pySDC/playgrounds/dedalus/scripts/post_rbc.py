#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 16:59:48 2025

@author: cpf5546
"""
import argparse

from pySDC.playgrounds.dedalus.problems.rbc import OutputFiles

parser = argparse.ArgumentParser(
    description='Post-process RBC Dedalus simulation',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "runDir", help="name of the simulation directory")

parser.add_argument(
    "--start", "-iB", help="first field index used",
    default=0, type=int)
parser.add_argument(
    "--stop", "-iE", help="index (+1) of the last field used",
    default=None, type=int)
parser.add_argument(
    "--step", "-iS", help="index step for the field used",
    default=1, type=int)

parser.add_argument(
    "--batchSize", "-b", help="number of fields post-processed simultaneously (all if not specified)",
    default=None, type=int)
parser.add_argument(
    "--verbose", "-v", help="print all post-processing operations in console",
    action="store_true")

args = parser.parse_args()

OutputFiles.VERBOSE = args.verbose

output = OutputFiles(args.runDir)

r = {"start": args.start, "stop": args.stop, "step": args.step}

output.getTimeSeries(which="all", batchSize=args.batchSize)
output.getProfiles(which="all", **r, batchSize=args.batchSize)
output.getSpectrum(which="all", **r, batchSize=args.batchSize)
