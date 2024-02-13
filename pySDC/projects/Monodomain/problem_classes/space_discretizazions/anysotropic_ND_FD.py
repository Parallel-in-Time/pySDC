import numpy as np
import scipy.sparse as sp

from pySDC.core.Errors import ProblemError
from pySDC.core.Problem import ptype
from pySDC.helpers import problem_helper
from pySDC.core.Common import RegisterParams


class AnysotropicNDimFinDiff(RegisterParams):
    def __init__(
        self,
        dom_size=[0.0, 1.0],
        nvars=512,
        diff=1.0,
        derivative=1,
        stencil_type='center',
        order=2,
        bc='periodic',
    ):
        # make sure parameters have the correct types
        if type(nvars) is not int:
            raise ProblemError('nvars should be int')

        if not type(diff) in [int, tuple]:
            raise ProblemError('coeff should be either tuple or int')

        if not type(dom_size) in [list, tuple]:
            raise ProblemError('dom_size should be list if dim==1 or tuple of lists else')

        if type(diff) is int:
            diff = (diff,)

        if type(dom_size) is list:
            dom_size = (dom_size,)

        # automatically determine ndim from dom_size
        self.ndim = len(dom_size)
        ndim = self.ndim
        if ndim > 3:
            raise ProblemError(f'can work with up to three dimensions, got {ndim}')

        if len(diff) != ndim:
            print(f'need len(diff)==dim, got {len(diff)}. I will repeat the first value for the other dimensions')
            diff = (diff[0],) * ndim

        # check for correct number of variables
        # if nvars % 2 != 0 and bc == 'periodic':
        #     raise ProblemError('the setup requires nvars % 2 == 0')
        # if (nvars + 1) % 2 != 0 and bc == 'dirichlet-zero':
        #     raise ProblemError('setup requires (nvars + 1) % 2 == 0')
        # if (nvars - 1) % 2 != 0 and bc == 'neumann-zero':
        #     raise ProblemError('setup requires (nvars - 1) % 2 == 0')

        # get maximal side of the domain
        dom_size_max = max([dom_size[i][1] - dom_size[i][0] for i in range(ndim)])
        # set nvars to be proportional to domain side
        nvars = [int(np.round(nvars * (dom_size[i][1] - dom_size[i][0]) / dom_size_max)) for i in range(ndim)]

        # compute dx and xvalues
        dx = []
        xvalues = []
        if bc == 'periodic':
            for i in range(ndim):
                dx.append((dom_size[i][1] - dom_size[i][0]) / nvars[i])
                xvalues.append(np.array([dom_size[i][0] + j * dx[i] for j in range(nvars[i])]))
        elif bc == 'dirichlet-zero':
            for i in range(ndim):
                dx.append((dom_size[i][1] - dom_size[i][0]) / (nvars[i] + 1))
                xvalues.append(np.array([dom_size[i][0] + (j + 1) * dx[i] for j in range(nvars[i])]))
        elif bc == 'neumann-zero':
            for i in range(ndim):
                dx.append((dom_size[i][1] - dom_size[i][0]) / (nvars[i] - 1))
                xvalues.append(np.array([dom_size[i][0] + j * dx[i] for j in range(nvars[i])]))
        else:
            raise ProblemError(f'Boundary conditions {bc} not implemented.')

        self.A = self.get_finite_difference_matrix(
            diff=diff,
            derivative=derivative,
            order=order,
            stencil_type=stencil_type,
            dx=dx,
            size=nvars,
            dim=ndim,
            bc=bc,
        )

        self.xvalues = xvalues
        self.dx = dx
        if ndim == 1:
            self.shape = (nvars[0],)
        elif ndim == 2:
            self.shape = (nvars[1], nvars[0])
        elif ndim == 3:
            self.shape = (nvars[2], nvars[1], nvars[0])

        self.Id = sp.eye(np.prod(nvars), format='csc')

        # store attribute and register them as parameters
        self._makeAttributeAndRegister('stencil_type', 'order', 'bc', localVars=locals(), readOnly=True)

    @property
    def grids(self):
        """ND grids associated to the problem"""
        x = self.xvalues
        if self.ndim == 1:
            return (x[0],)
        if self.ndim == 2:
            return x[0][None, :], x[1][:, None]
        if self.ndim == 3:
            return x[0][None, None, :], x[1][None, :, None], x[2][:, None, None]

    def get_finite_difference_matrix(self, diff, derivative, order, stencil_type=None, steps=None, dx=None, size=None, dim=None, bc=None):
        """
        Build FD matrix from stencils, with boundary conditions
        """

        if order > 2 and bc == 'dirichlet-zero':
            raise NotImplementedError('Higher order allowed only for periodic or neumann boundary conditions')

        # get stencil
        coeff, steps = problem_helper.get_finite_difference_stencil(derivative=derivative, order=order, stencil_type=stencil_type, steps=steps)

        A_1d = []
        if bc == 'dirichlet-zero':
            for j in range(dim):
                A_1d.append(sp.diags(coeff, steps, shape=(size[j], size[j]), format='csc'))
        elif bc == 'neumann-zero':
            for j in range(dim):
                A_1d.append(sp.diags(coeff, steps, shape=(size[j], size[j]), format='csc'))
                min_s = steps.min()
                for i in range(abs(min_s)):
                    for k in range(len(steps)):
                        if (i + steps[k]) < 0:
                            A_1d[j][i, i - steps[k]] = A_1d[j][i, i - steps[k]] + coeff[k]
                max_s = steps.max()
                for i in range(abs(max_s)):
                    for k in range(len(steps)):
                        if size[j] - 1 - i + steps[k] > size[j] - 1:
                            A_1d[j][size[j] - 1 - i, size[j] - 1 - i - steps[k]] = A_1d[j][size[j] - 1 - i, size[j] - 1 - i - steps[k]] + coeff[k]
        elif bc == 'periodic':
            for j in range(dim):
                A_1d.append(0 * sp.eye(size[j], format='csc'))
                for i in steps:
                    A_1d[j] += coeff[i] * sp.eye(size[j], k=steps[i])
                    if steps[i] > 0:
                        A_1d[j] += coeff[i] * sp.eye(size[j], k=-size[j] + steps[i])
                    if steps[i] < 0:
                        A_1d[j] += coeff[i] * sp.eye(size[j], k=size[j] + steps[i])
        else:
            raise NotImplementedError(f'Boundary conditions {bc} not implemented.')

        for k in range(dim):
            A_1d[k] /= dx[k] ** derivative

        if dim == 1:
            A = diff[0] * A_1d[0]
        elif dim == 2:
            A = diff[1] * sp.kron(A_1d[1], sp.eye(size[0])) + diff[0] * sp.kron(sp.eye(size[1]), A_1d[0])
        elif dim == 3:
            A = diff[2] * sp.kron(A_1d[2], sp.eye(size[1] * size[0])) + diff[0] * sp.kron(sp.eye(size[2] * size[1]), A_1d[0]) + diff[1] * sp.kron(sp.kron(sp.eye(size[2]), A_1d[1]), sp.eye(size[0]))
        else:
            raise NotImplementedError(f'Dimension {dim} not implemented.')

        return A
