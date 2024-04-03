import pytest


@pytest.mark.mpi4py
@pytest.mark.parametrize('name', ['imex_diffusion', 'imex_linear', 'mi_diffusion', 'mi_linear'])
@pytest.mark.parametrize('spectral', [True, False])
def test_GrayScottMPIFFT(name, spectral):
    """
    Test the implementation of the Gray-Scott problem by doing an Euler step forward and then an explicit Euler step
    backward to compute something akin to an error. We check that the "local error" has order 2.

    Keep
    """
    if name == 'imex_diffusion':
        from pySDC.implementations.problem_classes.GrayScott_MPIFFT import grayscott_imex_diffusion as problem_class
    elif name == 'imex_linear':
        from pySDC.implementations.problem_classes.GrayScott_MPIFFT import grayscott_imex_linear as problem_class
    elif name == 'mi_diffusion':
        from pySDC.implementations.problem_classes.GrayScott_MPIFFT import grayscott_mi_diffusion as problem_class
    elif name == 'mi_linear':
        from pySDC.implementations.problem_classes.GrayScott_MPIFFT import grayscott_mi_linear as problem_class
    import numpy as np

    prob = problem_class(spectral=spectral, nvars=(127,) * 2)

    dts = np.logspace(-3, -7, 15)
    errors = []

    for dt in dts:

        u0 = prob.u_exact(0)
        f0 = prob.eval_f(u0, 0)

        # do an IMEX or multi implicit Euler step forward
        if 'solve_system_2' in dir(prob):
            _u = prob.solve_system_1(u0, dt, u0, 0)
            u1 = prob.solve_system_2(_u, dt, _u, 0)
        else:
            u1 = prob.solve_system(u0 + dt * f0.expl, dt, u0, 0)

        # do an explicit Euler step backward
        f1 = prob.eval_f(u1, dt)
        u02 = u1 - dt * (np.sum(f1, axis=0))
        errors += [abs(u0 - u02)]

    errors = np.array(errors)
    dts = np.array(dts)
    order = np.log(errors[1:] / errors[:-1]) / np.log(dts[1:] / dts[:-1])
    mean_order = np.median(order)

    assert np.isclose(np.median(order), 2, atol=1e-2), f'Expected order 2, but got {mean_order}'
    assert prob.work_counters['rhs'].niter == len(errors) * 2
    if 'newton' in prob.work_counters.keys():
        assert prob.work_counters['newton'].niter > 0


if __name__ == '__main__':
    test_GrayScottMPIFFT('imex_diffusion', False)
