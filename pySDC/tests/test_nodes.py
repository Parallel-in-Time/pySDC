#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 11:58:05 2023

@author: cpf5546
"""
import pytest
import numpy as np

from pySDC.core.Nodes import NodesGenerator


def chebyNodes(kind, n):
    i = np.arange(n, dtype=float) + 1
    i = i[-1::-1]
    if kind == 1:
        nodes = np.cos((i - 0.5) / n * np.pi)
    elif kind == 2:
        nodes = np.cos(i / (n + 1) * np.pi)
    elif kind == 3:
        nodes = np.cos((i - 0.5) / (n + 0.5) * np.pi)
    elif kind == 4:
        nodes = np.cos(i / (n + 0.5) * np.pi)
    return tuple(nodes)


REF_NODES = {
    'LEGENDRE': {
        2: (-1 / 3**0.5, 1 / 3**0.5),
        3: (-((3 / 5) ** 0.5), 0, (3 / 5) ** 0.5),
        4: (
            -((3 / 7 + 2 / 7 * (6 / 5) ** 0.5) ** 0.5),
            -((3 / 7 - 2 / 7 * (6 / 5) ** 0.5) ** 0.5),
            (3 / 7 - 2 / 7 * (6 / 5) ** 0.5) ** 0.5,
            (3 / 7 + 2 / 7 * (6 / 5) ** 0.5) ** 0.5,
        ),
        5: (
            -1 / 3 * (5 + 2 * (10 / 7) ** 0.5) ** 0.5,
            -1 / 3 * (5 - 2 * (10 / 7) ** 0.5) ** 0.5,
            0,
            1 / 3 * (5 - 2 * (10 / 7) ** 0.5) ** 0.5,
            1 / 3 * (5 + 2 * (10 / 7) ** 0.5) ** 0.5,
        ),
    }
}

nTests = list(REF_NODES['LEGENDRE'].keys())
for kind in [1, 2, 3, 4]:
    REF_NODES[f'CHEBY-{kind}'] = {n: chebyNodes(kind, n) for n in nTests}


@pytest.mark.base
@pytest.mark.parametrize("node_type", REF_NODES.keys())
def test_nodesGeneration(node_type):
    gen = NodesGenerator(node_type=node_type, quad_type='GAUSS')
    ref = REF_NODES[node_type]
    for n, nodes in ref.items():
        assert np.allclose(nodes, gen.getNodes(n))
