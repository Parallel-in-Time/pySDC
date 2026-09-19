import numpy as np

from pySDC.core.errors import ProblemError
from pySDC.core.problem import Problem, WorkCounter
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


# noinspection PyUnusedLocal
class allencahn2d_imex(Problem):
    r"""
    Example implementing the two-dimensional Allen-Cahn equation with periodic boundary conditions, with the two
    phases at :math:`u = 0` and :math:`u = 1`

    .. math::
        \frac{\partial u}{\partial t} = \Delta u - \frac{2}{\varepsilon^2} u (1 - u)(1 - 2u)

    on a spatial domain :math:`[-\frac{L}{2}, \frac{L}{2}]^2`. Different initial conditions
    can be used, for example, circles of the form

    .. math::
        u({\bf x}, 0) = \frac{1}{2}\left(1 + \tanh\left(\frac{r - \sqrt{x_i^2 + y_j^2}}
        {\sqrt{2}\varepsilon}\right)\right),

    or *checker-board*

    .. math::
        u({\bf x}, 0) = \frac{1}{2}\left(1 + \sin(2 \pi x_i) \sin(2 \pi y_j)\right),

    or uniform distributed random numbers in :math:`[0, 1]` for :math:`i, j=0,..,N-1`, where :math:`N` is the number of
    spatial grid points. For time-stepping, the problem is treated *semi-implicitly*, i.e., the diffusion part is solved by
    Fast-Fourier Transform (FFT) and the nonlinear term is treated explicitly.

    An exact solution is not known, but instead the numerical solution can be compared via a generated reference solution computed
    by a ``SciPy`` routine.

    Parameters
    ----------
    nvars : List of int tuples, optional
        Number of unknowns in the problem, e.g. ``nvars=[(128, 128), (128, 128)]``.
    nu : int, optional
        Deprecated: only ``nu=2`` is supported, and anything else raises.
    eps : float, optional
        Scaling parameter :math:`\varepsilon`.
    radius : float, optional
        Radius of the circles.
    L : float, optional
        Denotes the period of the function to be approximated for the Fourier transform.
    init_type : str, optional
        Indicates which type of initial condition is used.
    useGPU : bool, optional
        Run on the GPU with CuPy instead of on the CPU with NumPy.

    Attributes
    ----------
    xvalues : np.1darray
        Grid points in space.
    dx : float
        Mesh width.
    lap : np.1darray
        Spectral operator for Laplacian.
    work_counters : WorkCounter
        Counts the right-hand side evaluations.
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    xp = np

    @classmethod
    def setup_GPU(cls):
        """
        Switch the array module and the datatypes over to CuPy.

        This changes the class, not the instance, as everything else in pySDC that does this
        does: once one instance of a class runs on the GPU, they all do.
        """
        import cupy as cp
        from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh, imex_cupy_mesh

        cls.xp = cp
        cls.dtype_u = cupy_mesh
        cls.dtype_f = imex_cupy_mesh

    def __init__(
        self,
        nvars=None,
        nu=2,
        eps=0.04,
        radius=0.25,
        L=1.0,
        init_type='circle',
        useGPU=False,
    ):
        """Initialization routine"""

        if useGPU:
            self.setup_GPU()

        if nvars is None:
            nvars = (128, 128)

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if len(nvars) != 2:
            raise ProblemError('this is a 2d example, got %s' % nvars)
        if nvars[0] != nvars[1]:
            raise ProblemError('need a square domain, got %s' % nvars)
        if nvars[0] % 2 != 0:
            raise ProblemError('the setup requires nvars = 2^p per dimension')

        if nu != 2:
            raise ProblemError(
                'the exponent nu is deprecated and only nu=2 is supported: the 0..1 form of Allen-Cahn '
                f'that this class now solves has no analogue of it, got nu={nu}'
            )

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(init=(nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'nvars', 'nu', 'eps', 'radius', 'L', 'init_type', 'useGPU', localVars=locals(), readOnly=True
        )

        self.dx = self.L / self.nvars[0]  # could be useful for hooks, too.
        self.xvalues = self.xp.array([i * self.dx - self.L / 2.0 for i in range(self.nvars[0])])

        kx = self.xp.zeros(self.init[0][0])
        ky = self.xp.zeros(self.init[0][1] // 2 + 1)

        kx[: int(self.init[0][0] / 2) + 1] = 2 * np.pi / self.L * self.xp.arange(0, int(self.init[0][0] / 2) + 1)
        kx[int(self.init[0][0] / 2) + 1 :] = (
            2 * np.pi / self.L * self.xp.arange(int(self.init[0][0] / 2) + 1 - self.init[0][0], 0)
        )
        ky[:] = 2 * np.pi / self.L * self.xp.arange(0, self.init[0][1] // 2 + 1)

        xv, yv = self.xp.meshgrid(kx, ky, indexing='ij')
        self.lap = -(xv**2) - yv**2

        self.work_counters['rhs'] = WorkCounter()

    def reaction(self, u):
        r"""The reaction term :math:`-\frac{2}{\varepsilon^2} u (1 - u)(1 - 2u)`."""
        return -2.0 / self.eps**2 * u * (1.0 - u) * (1.0 - 2.0 * u)

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """

        f = self.dtype_f(self.init)
        tmp = self.lap * self.xp.fft.rfft2(u)
        f.impl[:] = self.xp.fft.irfft2(tmp)
        if self.eps > 0:
            f.expl[:] = self.reaction(u)

        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple FFT solver for the diffusion part.

        Parameters
        ----------
        rhs  : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver (not used here so far).
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        me = self.dtype_u(self.init)

        tmp = self.xp.fft.rfft2(rhs) / (1.0 - factor * self.lap)
        me[:] = self.xp.fft.irfft2(tmp)

        return me

    def u_exact(self, t, u_init=None, t_init=None):
        r"""
        Routine to compute the exact solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.
        u_init : pySDC.implementations.problem_classes.allencahn2d_imex.dtype_u
            Initial conditions for getting the exact solution.
        t_init : float
            The starting time.

        Returns
        -------
        me : dtype_u
            The exact solution.
        """

        me = self.dtype_u(self.init, val=0.0)

        if t == 0:
            if self.init_type == 'circle':
                xv, yv = self.xp.meshgrid(self.xvalues, self.xvalues, indexing='ij')
                me[:, :] = 0.5 * (
                    1.0 + self.xp.tanh((self.radius - self.xp.sqrt(xv**2 + yv**2)) / (np.sqrt(2) * self.eps))
                )
            elif self.init_type == 'checkerboard':
                xv, yv = self.xp.meshgrid(self.xvalues, self.xvalues)
                me[:, :] = 0.5 * (1.0 + self.xp.sin(2.0 * np.pi * xv) * self.xp.sin(2.0 * np.pi * yv))
            elif self.init_type == 'random':
                me[:, :] = self.xp.random.uniform(0, 1, self.init)
            else:
                raise NotImplementedError('type of initial value not implemented, got %s' % self.init_type)
        else:

            def eval_rhs(t, u):
                f = self.eval_f(u.reshape(self.init[0]), t)
                return (f.impl + f.expl).flatten()

            me[:, :] = self.generate_scipy_reference_solution(eval_rhs, t, u_init, t_init)

        return me


class allencahn2d_imex_stab(allencahn2d_imex):
    r"""
    This implements the two-dimensional Allen-Cahn equation with periodic boundary conditions, with the two
    phases at :math:`u = 0` and :math:`u = 1`
    with stabilized splitting

    .. math::
        \frac{\partial u}{\partial t} = \Delta u - \frac{2}{\varepsilon^2} u (1 - u)(1 - 2u) + \frac{2}{\varepsilon^2}u

    on a spatial domain :math:`[-\frac{L}{2}, \frac{L}{2}]^2`. Different initial conditions
    can be used here, for example, circles of the form

    .. math::
        u({\bf x}, 0) = \frac{1}{2}\left(1 + \tanh\left(\frac{r - \sqrt{x_i^2 + y_j^2}}
        {\sqrt{2}\varepsilon}\right)\right),

    or *checker-board*

    .. math::
        u({\bf x}, 0) = \frac{1}{2}\left(1 + \sin(2 \pi x_i) \sin(2 \pi y_j)\right),

    or uniform distributed random numbers in :math:`[0, 1]` for :math:`i, j=0,..,N-1`, where :math:`N` is the number of
    spatial grid points. For time-stepping, the problem is treated *semi-implicitly*, i.e., the diffusion part is solved with
    Fast-Fourier Transform (FFT) and the nonlinear term is treated explicitly.

    An exact solution is not known, but instead the numerical solution can be compared via a generated reference solution computed
    by a ``SciPy`` routine.

    Parameters
    ----------
    nvars : List of int tuples, optional
        Number of unknowns in the problem, e.g. ``nvars=[(128, 128), (128, 128)]``.
    nu : int, optional
        Deprecated: only ``nu=2`` is supported, and anything else raises.
    eps : float, optional
        Scaling parameter :math:`\varepsilon`.
    radius : float, optional
        Radius of the circles.
    L : float, optional
        Denotes the period of the function to be approximated for the Fourier transform.
    init_type : str, optional
        Indicates which type of initial condition is used.
    useGPU : bool, optional
        Run on the GPU with CuPy instead of on the CPU with NumPy.

    Attributes
    ----------
    xvalues : np.1darray
        Grid points in space.
    dx : float
        Mesh width.
    lap : np.1darray
        Spectral operator for Laplacian.
    work_counters : WorkCounter
        Counts the right-hand side evaluations.
    """

    def __init__(self, nvars=None, nu=2, eps=0.04, radius=0.25, L=1.0, init_type='circle', useGPU=False):
        """Initialization routine"""

        if nvars is None:
            nvars = [(256, 256), (64, 64)]

        super().__init__(nvars, nu, eps, radius, L, init_type, useGPU)
        self.lap -= 2.0 / self.eps**2

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """

        f = self.dtype_f(self.init)
        tmp = self.lap * self.xp.fft.rfft2(u)
        f.impl[:] = self.xp.fft.irfft2(tmp)
        if self.eps > 0:
            f.expl[:] = self.reaction(u) + 2.0 / self.eps**2 * u

        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple FFT solver for the diffusion part.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver (not used here so far).
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        me = self.dtype_u(self.init)

        tmp = self.xp.fft.rfft2(rhs) / (1.0 - factor * self.lap)
        me[:] = self.xp.fft.irfft2(tmp)

        return me
