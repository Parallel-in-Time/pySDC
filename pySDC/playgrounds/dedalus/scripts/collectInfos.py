#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob
import shutil
import argparse

parser = argparse.ArgumentParser(
    description='Collect info files from simulation repositories',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "prefix", help="prefix of the simulation directories (is removed from the collected file names)")
parser.add_argument(
    "--outFolder", help="name of the output folder (default : prefix used)",
    default=None)
parser.add_argument(
    "--fileName", help="name of the collected file",
    default="infos.json")

args = parser.parse_args()

prefix = args.prefix
outFolder = prefix if args.outFolder is None else args.outFolder
fileName = parser.fileName

print(f"Collecting {prefix}*/{fileName} into {outFolder}")
os.makedirs(outFolder, exist_ok=True)

folders = glob.glob(args.prefix+"*")
for folder in folders:
    src = f"{folder}/{fileName}"
    if not os.path.isfile(src):
        print(f" -- {folder} does not contain {fileName}, ignoring ...")
        continue

    ext = fileName.split(".")[-1]
    dst = f"{outFolder}/{folder[len(prefix):]}.{ext}"
    print(f" -- copying {src} into {dst}")
    shutil.copy(src, dst)

print(" -- all done !")
