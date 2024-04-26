import pytest


def get_problem(nvars, xp, L):
    """
    Get a dummy problem to test interpolation

    Args:
        nvars (int): Number of DoF
        xp: Numerical library, i.e. Numpy or CuPy
        L (float): Length of space domain

    Returns:
        Instance of pySDC problem class
    """
    from pySDC.core.Problem import ptype
    from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh
    from pySDC.helpers.problem_helper import get_1d_grid

    class DummyProblem(ptype):
        dtype_u = mesh
        dtype_f = imex_mesh

        def __init__(self, nvars, spectral=False):
            super().__init__(init=(nvars, None, xp.dtype('float64')))
            self._makeAttributeAndRegister('nvars', 'spectral', localVars=locals())
            self.dx = L / (nvars + 1)
            self.x = xp.array([me * self.dx for me in range(nvars)])
            self.dx, self.x = get_1d_grid(nvars, 'periodic', right_boundary=L)

        def eval_f(self, *args, **kwargs):
            me = self.f_init
            me.impl[:] = xp.sin(2 * xp.pi / L * self.x)
            me.expl[:] = xp.cos(2 * xp.pi / L * self.x)
            return me

        def u_exact(self, t=0):
            me = self.u_init
            me[:] = xp.sin(2 * xp.pi / L * self.x)
            return me

    return DummyProblem(nvars)


def get_transfer_class(name):
    """
    Imports the problem class of the respective name and returns it.
    """
    if name == 'mesh_to_mesh_fft':
        from pySDC.implementations.transfer_classes.TransferMesh_FFT import mesh_to_mesh_fft as transfer_class
    elif name == 'mesh_to_mesh':
        from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh as transfer_class
    else:
        raise NotImplementedError(f'Don\'t know where to import transfer class {name!r}')
    return transfer_class


@pytest.mark.base
@pytest.mark.parametrize('name', ['mesh_to_mesh_fft', 'mesh_to_mesh'])
@pytest.mark.parametrize('order', [2, 4, 6, 8])
@pytest.mark.parametrize('L', [1.0])
def test_order(name, order, L):
    import numpy as xp

    params = {
        'iorder': order,
        'rorder': order,
        'periodic': True,
    }
    resolutions = [2**8, 2**7, 2**6, 2**5]

    errors = {
        'prolongation_u_exact': [],
        'restriction_u_exact': [],
        'prolongation_eval_f': [],
        'restriction_eval_f': [],
    }

    for function_name in ['u_exact', 'eval_f']:

        for res in resolutions[::-1]:
            fine = get_problem(res, xp, L)
            coarse = get_problem(res // 2, xp, L)
            transfer = get_transfer_class(name)(fine, coarse, params)

            fine_exact = fine.__getattribute__(function_name)(t=0)
            coarse_exact = coarse.__getattribute__(function_name)(t=0)

            errors[f'prolongation_{function_name}'] += [abs(fine_exact - transfer.prolong(coarse_exact))]
            errors[f'restriction_{function_name}'] += [abs(coarse_exact - transfer.restrict(fine_exact))]

    resolutions = xp.array(resolutions)
    for key, value in errors.items():
        value = xp.array(value)

        mask = value > 1e-15

        if mask.any():
            num_order = abs(
                xp.log(value[mask][1:] / value[mask][:-1]) / xp.log(resolutions[mask][1:] / resolutions[mask][:-1])
            )
            assert xp.isclose(
                xp.median(num_order), order, atol=0.3
            ), f'Got unexpected order {xp.median(num_order):.2f} when expecting {order} in {key}. Errors: {value}'


if __name__ == '__main__':
    test_order('mesh_to_mesh', 8, 1.0)
