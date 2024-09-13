#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 09:53:34 2024

Testing scripts for the parallel_SDC_reloaded project
"""
import sys
import pytest


@pytest.mark.base
@pytest.mark.parametrize(
    "sName",
    [
        "convergence",
        "nilpotency",
        "stability",
    ],
)
def test_script(sName):
    try:
        exec(f"from pySDC.projects.parallelSDC_reloaded import {sName}")
    except Exception as e:
        raise ImportError(f"error when executing {sName}.py : {e}")


@pytest.mark.base
@pytest.mark.parametrize("sType", ["setup", "accuracy"])
@pytest.mark.parametrize(
    "pName",
    [
        "allenCahn",
        "chemicalReaction",
        "jacobiElliptic",
        "kaps",
        "lorenz",
        "protheroRobinson",
        "protheroRobinsonAutonomous",
        "vanderpol",
    ],
)
def test_playgrounds(pName, sType):
    try:
        exec(f"from pySDC.projects.parallelSDC_reloaded import {pName}_{sType}")
    except Exception as e:
        raise ImportError(f"error when executing {pName}_{sType}.py : {e}")


@pytest.mark.base
def test_script_fig01_conv():
    from pySDC.projects.parallelSDC_reloaded.scripts import fig01_conv

    assert fig01_conv.config == [
        (4, "RADAU-RIGHT", "MIN-SR-NS"),
        (5, "LOBATTO", "MIN-SR-NS"),
        (4, "RADAU-RIGHT", "MIN-SR-S"),
        (5, "LOBATTO", "MIN-SR-S"),
        (4, "RADAU-RIGHT", "MIN-SR-FLEX"),
        (5, "LOBATTO", "MIN-SR-FLEX"),
    ]


@pytest.mark.base
def test_script_fig02_stab():
    from pySDC.projects.parallelSDC_reloaded.scripts import fig02_stab

    assert fig02_stab.config == [
        "PIC",
        "MIN-SR-NS",
        "MIN-SR-S",
        "MIN-SR-FLEX",
        "LU",
        "VDHS",
    ]


@pytest.mark.base
def test_script_fig03_lorenz():
    from pySDC.projects.parallelSDC_reloaded.scripts import fig03_lorenz

    minPrec = fig03_lorenz.minPrec
    assert minPrec == ["MIN-SR-NS", "MIN-SR-S", "MIN-SR-FLEX"]
    assert fig03_lorenz.config == [
        [(*minPrec, "LU", "EE", "PIC"), 4],
        [(*minPrec, "VDHS", "RK4", "ESDIRK43"), 4],
        [(*minPrec, "PIC", "RK4", "ESDIRK43"), 5],
    ]


@pytest.mark.base
def test_script_fig04_protheroRobinson():
    from pySDC.projects.parallelSDC_reloaded.scripts import fig04_protheroRobinson

    minPrec = fig04_protheroRobinson.minPrec
    assert minPrec == ["MIN-SR-NS", "MIN-SR-S", "MIN-SR-FLEX"]
    assert fig04_protheroRobinson.config == [
        [(*minPrec, "VDHS", "ESDIRK43", "LU"), 4],
        [(*minPrec, "VDHS", "ESDIRK43", "LU"), 6],
    ]


@pytest.mark.base
def test_script_fig05_allenCahn():
    # Test fails for python < 3.8, so avoid it
    if sys.version_info.minor < 8:
        return

    from pySDC.projects.parallelSDC_reloaded.scripts import fig05_allenCahn

    minPrec = fig05_allenCahn.minPrec
    assert minPrec == ["MIN-SR-NS", "MIN-SR-S", "MIN-SR-FLEX"]
    assert fig05_allenCahn.config == [
        (*minPrec, "VDHS", "ESDIRK43", "LU"),
    ]


@pytest.mark.base
def test_script_fig06_allenCahn():
    # Test fails for python < 3.8, so avoid it
    if sys.version_info.minor < 8:
        return

    from pySDC.projects.parallelSDC_reloaded.scripts import fig06_allenCahnMPI, fig06_allenCahnMPI_plot

    assert fig06_allenCahnMPI.nSweeps == 4
    assert fig06_allenCahnMPI_plot.minPrec == ["MIN-SR-NS", "MIN-SR-S", "MIN-SR-FLEX"]
