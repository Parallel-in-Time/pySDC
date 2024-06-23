#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 17:17:14 2023

Evaluate the nilpotency of diagonal preconditionners MIN-SR-S and MIN-SR-NS
with increasing number of nodes.
"""
import numpy as np
import matplotlib.pyplot as plt

from pySDC.core.sweeper import Sweeper

quadType = "LOBATTO"
nodeType = "LEGENDRE"


def nilpotencyS(d, Q):
    if quadType in ['LOBATTO', 'RADAU-LEFT']:
        d = d[1:]
        Q = Q[1:, 1:]
    M = d.size
    D = np.diag(1 / d)
    K = np.eye(M) - D @ Q
    return np.linalg.norm(np.linalg.matrix_power(K, M), ord=np.inf)


def nilpotencyNS(d, Q):
    M = d.size
    D = np.diag(d)
    K = D - Q
    return np.linalg.norm(np.linalg.matrix_power(K, M), ord=np.inf)


nil_MIN_SR_S = []
nil_MIN_SR_NS = []
nNodes = range(2, 20)
for m in nNodes:
    s = Sweeper({"num_nodes": m, "quad_type": quadType, "node_type": nodeType})
    Q = s.coll.Qmat[1:, 1:]
    nodes = s.coll.nodes

    qDelta = s.get_Qdelta_implicit(qd_type="MIN-SR-S")
    d = np.diag(qDelta)[1:]
    nil_MIN_SR_S.append([nilpotencyS(d, Q), nilpotencyNS(d, Q)])

    qDelta = s.get_Qdelta_implicit(qd_type="MIN-SR-NS")
    d = np.diag(qDelta)[1:]
    nil_MIN_SR_NS.append([nilpotencyS(d, Q), nilpotencyNS(d, Q)])

nil_MIN_SR_NS = np.array(nil_MIN_SR_NS).T
nil_MIN_SR_S = np.array(nil_MIN_SR_S).T

fig = plt.figure("nilpotency")
nNodes = np.array(nNodes, np.float128)
plt.semilogy(nNodes, nil_MIN_SR_NS[0], 'o-', label="MIN-SR-NS (nill. stiff)")
plt.semilogy(nNodes, nil_MIN_SR_NS[1], 'o--', label="MIN-SR-NS (nill. non-stiff)")
plt.semilogy(nNodes, nil_MIN_SR_S[0], 's-', label="MIN-SR-S (nill. stiff)")
plt.semilogy(nNodes, nil_MIN_SR_S[1], 's--', label="MIN-SR-S (nill. non-stiff)")
plt.legend()
plt.semilogy(nNodes, 14**nNodes * 1e-17, ':', c="gray")
plt.grid(True)
plt.xlabel("M")
plt.ylabel("nilpotency")
fig.set_size_inches(8.65, 5.33)
plt.tight_layout()
