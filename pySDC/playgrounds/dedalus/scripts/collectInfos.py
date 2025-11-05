#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob
import shutil
import argparse

parser = argparse.ArgumentParser(
    description='Collect simulation infos from simulation repositories',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "prefix", help="prefix of the simulation directories (is removed from the collected file names)")
parser.add_argument(
    "--outFolder", help="name of the output folder (default : prefix used)",
    default=None)

args = parser.parse_args()

prefix = args.prefix
outFolder = prefix if args.outFolder is None else args.outFolder

print(f"Collecting {prefix}*/infos.json into {outFolder}")
os.makedirs(outFolder, exist_ok=True)

folders = glob.glob(args.prefix+"*")
for folder in folders:
    src = f"{folder}/infos.json"
    if not os.path.isfile(src):
        print(f" -- {folder} does not contain infos.json, ignoring ...")
        continue

    dst = f"{outFolder}/{folder[len(prefix):]}.json"
    print(f" -- copying {src} into {dst}")
    shutil.copy(src, dst)

print(" -- all done !")    