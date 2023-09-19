import numpy as np

from pySDC.core.Errors import ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


# noinspection PyUnusedLocal
class allencahn2d_imex(ptype):
    r"""
    Example implementing the two-dimensional Allen-Cahn equation with periodic boundary conditions :math:`u \in [-1, 1]^2`

    .. math::
        \frac{\partial u}{\partial t} = \Delta u + \frac{1}{\varepsilon^2} u (1 - u^\nu)

    on a spatial domain :math:`[-\frac{L}{2}, \frac{L}{2}]^2`, and constant parameter :math:`\nu`. Different initial conditions
    can be used, for example, circles of the form

    .. math::
        u({\bf x}, 0) = \tanh\left(\frac{r - \sqrt{x_i^2 + y_j^2}}{\sqrt{2}\varepsilon}\right),

    or *checker-board*

    .. math::
        u({\bf x}, 0) = \sin(2 \pi x_i) \sin(2 \pi y_j),

    or uniform distributed random numbers in :math:`[-1, 1]` for :math::`i, j=0,..,N-1`, where :math:`N` is the number of
    spatial grid points. For time-stepping, the problem is treated *fully-implicitly*, i.e., the nonlinear system is solved by
    Fast-Fourier Tranform (FFT).

    An exact solution is not known, but instead the numerical solution can be compared via a generated reference solution computed
    by a scipy routine.

    Parameters
    ----------
    nvars : int
        Number of unknowns in the problem.
    nu : float
        Problem parameter.
    eps : float
        Problem parameter.
    radius : float
        Radius of the circles.
    L : int
        Denotes the period of the function to be approximated for the Fourier transform.
    init_type : str
        Indicates which type of initial condition is used.

    Attributes
    ----------
    xvalues : np.ndarray
        Grid points in space.
    dx : float
        Mesh width.
    lap : np.ndarray
        Spectral operator for Laplacian.
    rfft_object :
        Planned real FFT for forward transformation.
    irfft_object :
        Planned IFFT for backward transformation.
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(self, nvars=None, nu=2, eps=0.04, radius=0.25, L=1.0, init_type='circle'):
        """Initialization routine"""

        if nvars is None:
            nvars = [(256, 256), (64, 64)]

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if len(nvars) != 2:
            raise ProblemError('this is a 2d example, got %s' % nvars)
        if nvars[0] != nvars[1]:
            raise ProblemError('need a square domain, got %s' % nvars)
        if nvars[0] % 2 != 0:
            raise ProblemError('the setup requires nvars = 2^p per dimension')

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(init=(nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'nvars', 'nu', 'eps', 'radius', 'L', 'init_type', localVars=locals(), readOnly=True
        )

        self.dx = self.L / self.nvars[0]  # could be useful for hooks, too.
        self.xvalues = np.array([i * self.dx - self.L / 2.0 for i in range(self.nvars[0])])

        kx = np.zeros(self.init[0][0])
        ky = np.zeros(self.init[0][1] // 2 + 1)

        kx[: int(self.init[0][0] / 2) + 1] = 2 * np.pi / self.L * np.arange(0, int(self.init[0][0] / 2) + 1)
        kx[int(self.init[0][0] / 2) + 1 :] = (
            2 * np.pi / self.L * np.arange(int(self.init[0][0] / 2) + 1 - self.init[0][0], 0)
        )
        ky[:] = 2 * np.pi / self.L * np.arange(0, self.init[0][1] // 2 + 1)

        xv, yv = np.meshgrid(kx, ky, indexing='ij')
        self.lap = -(xv**2) - yv**2

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
        v = u.flatten()
        tmp = self.lap * np.fft.rfft2(u)
        f.impl[:] = np.fft.irfft2(tmp)
        if self.eps > 0:
            f.expl[:] = (1.0 / self.eps**2 * v * (1.0 - v**self.nu)).reshape(self.nvars)
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

        tmp = np.fft.rfft2(rhs) / (1.0 - factor * self.lap)
        me[:] = np.fft.irfft2(tmp)

        return me

    def u_exact(self, t, u_init=None, t_init=None):
        """
        Routine to compute the exact solution at time t.

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
                xv, yv = np.meshgrid(self.xvalues, self.xvalues, indexing='ij')
                me[:, :] = np.tanh((self.radius - np.sqrt(xv**2 + yv**2)) / (np.sqrt(2) * self.eps))
            elif self.init_type == 'checkerboard':
                xv, yv = np.meshgrid(self.xvalues, self.xvalues)
                me[:, :] = np.sin(2.0 * np.pi * xv) * np.sin(2.0 * np.pi * yv)
            elif self.init_type == 'random':
                me[:, :] = np.random.uniform(-1, 1, self.init)
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
    This implements the two-dimensional Allen-Cahn equation with periodic boundary conditions :math:`u \in [-1, 1]^2`
    with stabilized splitting

    .. math::
        \frac{\partial u}{\partial t} = \Delta u + \frac{1}{\varepsilon^2} u (1 - u^\nu) + \frac{2}{\varepsilon^2}u

    on a spatial domain :math:`[-\frac{L}{2}, \frac{L}{2}]^2`, and constant parameter :math:`\nu`. Different initial conditions
    can be used here, for example, circles of the form

    .. math::
        u({\bf x}, 0) = \tanh\left(\frac{r - \sqrt{x_i^2 + y_j^2}}{\sqrt{2}\varepsilon}\right),

    or *checker-board*

    .. math::
        u({\bf x}, 0) = \sin(2 \pi x_i) \sin(2 \pi y_j),

    or uniform distributed random numbers in :math:`[-1, 1]` for :math::`i, j=0,..,N-1`, where :math:`N` is the number of
    spatial grid points. For time-stepping, the problem is treated *fully-implicitly*, i.e., the nonlinear system is solved by
    Fast-Fourier Tranform (FFT).

    An exact solution is not known, but instead the numerical solution can be compared via a generated reference solution computed
    by a scipy routine.

    Parameters
    ----------
    nvars : int
        Number of unknowns in the problem.
    nu : float
        Problem parameter.
    eps : float
        Problem parameter.
    radius : float
        Radius of the circles.
    L : int
        Denotes the period of the function to be approximated for the Fourier transform.
    init_type : str
        Indicates which type of initial condition is used.

    Attributes
    ----------
    xvalues : np.ndarray
        Grid points in space.
    dx : float
        Mesh width.
    lap : np.ndarray
        Spectral operator for Laplacian.
    rfft_object :
        Planned real FFT for forward transformation.
    irfft_object :
        Planned IFFT for backward transformation.
    """

    def __init__(self, nvars=None, nu=2, eps=0.04, radius=0.25, L=1.0, init_type='circle'):
        """Initialization routine"""

        if nvars is None:
            nvars = [(256, 256), (64, 64)]

        super().__init__(nvars, nu, eps, radius, L, init_type)
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
        v = u.flatten()
        tmp = self.lap * np.fft.rfft2(u)
        f.impl[:] = np.fft.irfft2(tmp)
        if self.eps > 0:
            f.expl[:] = (1.0 / self.eps**2 * v * (1.0 - v**self.nu) + 2.0 / self.eps**2 * v).reshape(self.nvars)
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

        tmp = np.fft.rfft2(rhs) / (1.0 - factor * self.lap)
        me[:] = np.fft.irfft2(tmp)

        return me
