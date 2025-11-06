#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Investigate the load of each compute node
"""
import pandas as pd

decompFile = "_decomp_N32.txt"

nodes = {}

with open(decompFile, "r") as f:
    lines = f.readlines()

lines = [l.strip() for l in lines[4:]]

nProcs = len(lines) // 5
for i in range(nProcs):
    pos = 5*i
    ranks = lines[pos].split()[0]
    gR, sR, tR = ranks.split("-")
    cpu, nodeName = lines[pos+1][3:].split(" on ")
    cpu = cpu.replace(" ", "").replace("}", "").replace("{", "")

    try:
        nodes[nodeName][cpu] = (gR, sR, tR)
    except:
        nodes[nodeName] = {cpu: (gR, sR, tR)}


load = {}
for name in nodes:
    cpus = nodes[name]
    nCPUs = len(cpus.keys())

    tGroups = set(g[-1] for g in cpus.values())
    sGroups = set(g[-2] for g in cpus.values())

    load[name] = dict(
        nCPUs=nCPUs,
        nTGroups=len(tGroups), nSGroups=len(sGroups),
        tGroups=tGroups, sGroups=sGroups,
        )

load = pd.DataFrame(load)
print(load.iloc[:3])
