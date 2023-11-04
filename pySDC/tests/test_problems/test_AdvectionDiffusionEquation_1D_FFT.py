import pytest


def get_error_thresholds(freq, nu):
    r"""
    Returns the error thresholds for parameters ``nu`` and ``freq``.

    Parameters
    ----------
    freq : int
        Wave number.
    nu : float
        Diffusion coefficient.
    """
    e_tol_imex = {
        -1: {
            0.02: 0.011,
        },
        0: {
            -0.02: 0.076,
            0.02: 0.055,
        },
        1: {
            -0.02: 0.0063,
            0.02: 0.0063,
        },
    }

    e_tol_full = {
        -1: {
            0.02: 0.00021,
        },
        0: {
            -0.02: 0.078,
            0.02: 0.064,
        },
        1: {
            -0.02: 2.01e-05,
            0.02: 2e-05,
        },
    }
    return e_tol_imex[freq][nu], e_tol_full[freq][nu]


@pytest.mark.base
@pytest.mark.parametrize('freq', [-1, 0, 1])
@pytest.mark.parametrize('nu', [0.02, -0.02])
def test_imex_vs_implicit(freq, nu):
    import numpy as np
    from pySDC.core.Errors import ParameterError, ProblemError
    from pySDC.implementations.problem_classes.AdvectionDiffusionEquation_1D_FFT import (
        advectiondiffusion1d_imex,
        advectiondiffusion1d_implicit,
    )

    problem_params = {
        'nvars': 32,
        'c': 1.0,
        'freq': freq,
        'nu': nu,
        'L': 1.0,
    }

    imex = advectiondiffusion1d_imex(**problem_params)
    fully_impl = advectiondiffusion1d_implicit(**problem_params)

    t0 = 0.0
    if freq < 0 and nu < 0:
        # check if ParameterError is raised correctly for freq < 0 and nu < 0
        with pytest.raises(ParameterError):
            imex.u_exact(t0)
    else:
        # test if evaluations of right-hand side do match
        u0 = imex.u_exact(t0)

        tmp = imex.eval_f(u0, t0)
        f_imex = tmp.expl + tmp.impl
        f_full = fully_impl.eval_f(u0, t0)

        assert np.allclose(
            f_imex, f_full
        ), 'Evaluation of right-hand side in semi-explicit case and in fully-implicit case do not match!'

        # test if solving one time step satisfies a specific error threshold
        dt = 1e-3
        args = {
            'rhs': u0,
            'factor': dt,
            'u0': u0,
            't': t0,
        }

        u_ex = imex.u_exact(t0 + dt)

        u_imex = imex.solve_system(**args)
        u_full = fully_impl.solve_system(**args)

        e_imex = abs(u_ex - u_imex)
        e_full = abs(u_ex - u_full)

        e_tol_imex, e_tol_full = get_error_thresholds(freq, nu)

        assert e_imex < e_tol_imex, "Error is too large in semi-explicit case!"
        assert e_full < e_tol_full, "Error is too large in fully-implicit case!"

        # check if ProblemError is raised correctly in case if nvars % 2 != 0
        problem_params.update({'nvars': 31})
        with pytest.raises(ProblemError):
            imex_test = advectiondiffusion1d_imex(**problem_params)
            fully_impl_test = advectiondiffusion1d_implicit(**problem_params)
