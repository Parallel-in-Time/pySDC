import pytest


@pytest.mark.petsc
@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize('variant', ['multiimplicit', 'fullyimplicit', 'semiimplicit'])
def test_eval_f_in_parallel(variant):
    """
    eval_f zeroes the two boundary points and leaves the rest alone, also when they live on other ranks.
    """
    import numpy as np
    from petsc4py import PETSc
    import pySDC.implementations.problem_classes.GeneralizedFisher_1D_PETSc as module

    prob = getattr(module, f'petsc_fisher_{variant}')(
        nvars=31, lambda0=2.0, nu=1, interval=(-5, 5), lsol_maxiter=100, nlsol_maxiter=100
    )
    u = prob.u_exact(0.1)
    f = prob.eval_f(u, 0.1)

    x = u.getArray()
    diffusion = (prob.A * u).getArray()
    reaction = prob.lambda0**2 * x * (1 - x**prob.nu)
    if variant == 'fullyimplicit':
        parts = [(f, diffusion + reaction)]
    else:
        parts = zip([getattr(f, name) for name in f.components], [diffusion, reaction], strict=True)
    for vec, ref in parts:
        ref = ref.copy()
        if prob.xs == 0:
            ref[0] = 0
        if prob.xe == prob.nvars:
            ref[-1] = 0
        assert np.allclose(vec.getArray(), ref, rtol=1e-14, atol=0), f'{variant} on rank {PETSc.COMM_WORLD.rank}'
