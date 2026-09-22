import pytest


def get_problem(nvars, xp, L, mpifft=False, spectral=False, x0=0, useGPU=False, ndim=2):
    """
    Get a dummy problem to test interpolation

    Args:
        nvars (int): Number of DoF
        xp: Numerical library, i.e. Numpy or CuPy
        L (float): Length of space domain
        ndim (int): Number of spatial dimensions, for the MPI-FFT problem only

    Returns:
        Instance of pySDC problem class
    """
    from pySDC.core.problem import Problem
    from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh
    from pySDC.helpers.problem_helper import get_1d_grid

    class DummyProblem(Problem):
        dtype_u = mesh
        dtype_f = imex_mesh

        def __init__(self, nvars):
            super().__init__(init=(nvars, None, xp.dtype('float64')))
            self._makeAttributeAndRegister('nvars', localVars=locals())
            self.dx = L / (nvars + 1)
            self.x = xp.array([x0 + me * self.dx for me in range(nvars)])
            self.dx, self.x = get_1d_grid(nvars, 'periodic', right_boundary=L)

        def eval_f(self, *args, **kwargs):
            me = self.f_init
            me.impl[:] = xp.sin(2 * xp.pi / L * self.x)
            me.expl[:] = xp.cos(2 * xp.pi / L * self.x)
            return me

        def u_exact(self, *args, **kwargs):
            me = self.u_init
            me[:] = xp.sin(2 * xp.pi / L * self.x)
            return me

    if mpifft:
        from pySDC.implementations.problem_classes.generic_MPIFFT_Laplacian import IMEX_Laplacian_MPIFFT

        class DummyProblemMPIFFT(IMEX_Laplacian_MPIFFT):
            def _fill(self):
                me = xp.sin(2 * xp.pi / L * self.X[0])
                if self.spectral:
                    return self.fft.forward(me)
                else:
                    return me

            def eval_f(self, *args, **kwargs):
                me = self.f_init
                me.impl[:] = self._fill()
                me.expl[:] = self._fill()
                return me

            def u_exact(self, *args, **kwargs):
                me = self.u_init
                me[:] = self._fill()
                return me

        return DummyProblemMPIFFT(nvars=(nvars,) * ndim, spectral=spectral, x0=x0, L=L, useGPU=useGPU)

    return DummyProblem(nvars)


def get_transfer_class(name):
    """
    Imports the problem class of the respective name and returns it.
    """
    if name == 'mesh_to_mesh_fft':
        from pySDC.implementations.transfer_classes.TransferMesh_FFT import mesh_to_mesh_fft as transfer_class
    elif name == 'mesh_to_mesh':
        from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh as transfer_class
    elif name == 'fft_to_fft':
        from pySDC.implementations.transfer_classes.TransferMesh_MPIFFT import fft_to_fft as transfer_class
    else:
        raise NotImplementedError(f'Don\'t know where to import transfer class {name!r}')
    return transfer_class


@pytest.mark.base
@pytest.mark.parametrize('L', [1.0, 6.283185307179586])
@pytest.mark.parametrize('x0', [0.0, -3.141592653589793])
def test_mesh_to_mesh_fft(L, x0):
    single_test('mesh_to_mesh_fft', -1.0, L, mpifft=False, x0=x0)


@pytest.mark.base
@pytest.mark.parametrize('order', [2, 4, 6, 8])
@pytest.mark.parametrize('L', [1.0, 6.283185307179586])
@pytest.mark.parametrize('x0', [0.0, -3.141592653589793])
def test_mesh_to_mesh(order, L, x0):
    single_test('mesh_to_mesh', order, L, mpifft=False, x0=x0)


@pytest.mark.mpi4py
@pytest.mark.parametrize('L', [1.0, 6.283185307179586])
@pytest.mark.parametrize('spectral', [True, False])
@pytest.mark.parametrize('x0', [0.0, -3.141592653589793])
def test_fft_to_fft(L, spectral, x0):
    single_test('fft_to_fft', -1, L, mpifft=True, spectral=spectral, x0=x0)


@pytest.mark.mpi4py
@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('spectral', [True, False])
def test_fft_to_fft_dimensions(ndim, spectral):
    """
    The restriction in `fft_to_fft` used to slice with a hardcoded two dimensions. Prolongation
    followed by restriction has to give back the coarse data in any number of dimensions.
    """
    import numpy as xp

    L = 2 * xp.pi
    fine = get_problem(16, xp, L, mpifft=True, spectral=spectral, ndim=ndim)
    coarse = get_problem(8, xp, L, mpifft=True, spectral=spectral, ndim=ndim)
    transfer = get_transfer_class('fft_to_fft')(fine, coarse, {})

    for function_name in ['u_exact', 'eval_f']:
        u = coarse.__getattribute__(function_name)(t=0)
        assert (
            abs(transfer.restrict(transfer.prolong(u)) - u) < 1e-12
        ), f'Restriction does not undo prolongation of {function_name} in {ndim}d'


@pytest.mark.mpi4py
@pytest.mark.parametrize('spectral', [True, False])
def test_fft_to_fft_component_axis(spectral):
    """
    Problems that declare `ncomp` keep their components in the first axis (Gray-Scott) or in the last
    one (the temperature-coupled Allen-Cahn). Either way each component has to come out exactly as a
    single-component problem of the same resolution would.
    """
    import numpy as xp
    from pySDC.implementations.problem_classes.generic_MPIFFT_Laplacian import IMEX_Laplacian_MPIFFT
    from pySDC.implementations.problem_classes.GrayScott_MPIFFT import grayscott_imex_diffusion

    nvars_fine, nvars_coarse = (32, 32), (16, 16)
    fine = grayscott_imex_diffusion(nvars=nvars_fine, spectral=spectral)
    coarse = grayscott_imex_diffusion(nvars=nvars_coarse, spectral=spectral)
    assert fine.u_exact(0).shape[0] == fine.ncomp, 'Expected Gray-Scott to keep its components in the first axis'
    transfer = get_transfer_class('fft_to_fft')(fine, coarse, {})

    # the same transfer, but for a problem with a single component
    ref_fine = IMEX_Laplacian_MPIFFT(nvars=nvars_fine, spectral=spectral)
    ref_coarse = IMEX_Laplacian_MPIFFT(nvars=nvars_coarse, spectral=spectral)
    ref_transfer = get_transfer_class('fft_to_fft')(ref_fine, ref_coarse, {})

    for direction, source, ref_source in [('restrict', fine, ref_fine), ('prolong', coarse, ref_coarse)]:
        u = source.u_exact(0)
        transferred = getattr(transfer, direction)(u)

        for i in range(fine.ncomp):
            single = ref_source.u_init
            single[:] = u[i]
            assert xp.allclose(
                transferred[i], getattr(ref_transfer, direction)(single)
            ), f'Component {i} of the {direction}ion does not match the single-component problem'


@pytest.mark.cupy
@pytest.mark.parametrize('L', [1.0, 6.283185307179586])
@pytest.mark.parametrize('spectral', [True, False])
@pytest.mark.parametrize('x0', [0.0, -3.141592653589793])
def test_fft_to_fft_GPU(L, spectral, x0):
    single_test('fft_to_fft', -1, L, mpifft=True, spectral=spectral, x0=x0, useGPU=True)


def single_test(name, order, L, mpifft, spectral=False, x0=0, useGPU=False):
    import numpy as xp

    params = {
        'iorder': order,
        'rorder': order,
        'periodic': True,
    }
    resolutions = [2**8, 2**7, 2**6, 2**5]

    errors = {
        'restriction_u_exact': [],
        'restriction_eval_f': [],
        'prolongation_u_exact': [],
        'prolongation_eval_f': [],
    }

    for function_name in ['u_exact', 'eval_f']:
        for res in resolutions[::-1]:
            fine = get_problem(res, xp, L, mpifft=mpifft, spectral=spectral, x0=x0, useGPU=useGPU)
            coarse = get_problem(res // 2, xp, L, mpifft=mpifft, spectral=spectral, x0=x0, useGPU=useGPU)
            transfer = get_transfer_class(name)(fine, coarse, params)

            fine_exact = fine.__getattribute__(function_name)(t=0)
            coarse_exact = coarse.__getattribute__(function_name)(t=0)

            errors[f'prolongation_{function_name}'] += [abs(fine_exact - transfer.prolong(coarse_exact))]
            errors[f'restriction_{function_name}'] += [abs(coarse_exact - transfer.restrict(fine_exact))]

    resolutions = xp.array(resolutions)
    for key, value in errors.items():
        value = xp.array(value)

        mask = value > 1e-14

        if mask.any():
            num_order = abs(
                xp.log(value[mask][1:] / value[mask][:-1]) / xp.log(resolutions[mask][1:] / resolutions[mask][:-1])
            )
            assert xp.isclose(
                xp.median(num_order), order, atol=0.3
            ), f'Got unexpected order {xp.median(num_order):.2f} when expecting {order} in {key}. Errors: {value}'


if __name__ == '__main__':
    import numpy as np

    test_fft_to_fft('fft_to_fft', 2, 2 * np.pi, True, 0.0)
