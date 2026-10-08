"""
Problem classes that are built with their default resolution.

A list of resolutions, one per level, is what a description passes and what the step splits up, so a problem class
only ever receives one of its entries. These classes had such a list as their own default, and failed to build.
"""

import pytest


@pytest.mark.base
def test_allencahn2d_imex_stab():
    from pySDC.implementations.problem_classes.AllenCahn_2D_FFT import allencahn2d_imex_stab

    assert allencahn2d_imex_stab().nvars == (256, 256)


@pytest.mark.mpi4py
def test_allencahn_temp_imex():
    from pySDC.implementations.problem_classes.AllenCahn_Temp_MPIFFT import allencahn_temp_imex

    assert allencahn_temp_imex().nvars == (128, 128)


@pytest.mark.fenics
def test_fenics_vortex_2d():
    from pySDC.implementations.problem_classes.VorticityVelocity_2D_FEniCS_periodic import fenics_vortex_2d

    assert fenics_vortex_2d().c_nvars == (32, 32)
