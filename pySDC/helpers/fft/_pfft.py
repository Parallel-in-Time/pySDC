"""The distributed transform: serial transforms on the local axes, transposes in between."""

import numpy as np
from mpi4py_fft.mpifft import PFFT as _PFFT
from mpi4py_fft.mpifft import Transform
from mpi4py_fft.pencil import Subcomm

from pySDC.helpers.fft._serial import CuPyFFT
from pySDC.helpers.fft._transfer import CuPyPencil


class PFFT_GPU(_PFFT):
    """A :class:`mpi4py_fft.mpifft.PFFT` whose data stays on the GPU.

    Everything :class:`~mpi4py_fft.mpifft.PFFT` computes after construction -- shapes, local
    slices, dtypes, and the chaining of transforms with transposes -- is inherited. Only
    ``__init__`` is restated, because it names the classes ``Pencil`` and ``FFT`` directly and
    there is no way in from outside to substitute the GPU versions of them.

    Kept deliberately close to the original so that the two can be compared line by line when
    ``mpi4py-fft`` changes, which at the time of writing it has not done in over a year.
    """

    #: read by `newDistArray` to decide which kind of array to hand back
    on_GPU = True

    def __init__(
        self,
        comm,
        shape=None,
        axes=None,
        dtype=float,
        grid=None,
        padding=False,
        collapse=False,
        backend='cupyx-scipy',
        transforms=None,
        darray=None,
        comm_backend='NCCL',
        **kw,
    ):
        assert darray is None, 'PFFT_GPU is always given a shape rather than an existing array'
        assert shape is not None

        axes = self._normalise_axes(axes, shape)
        self.axes = axes
        shape = list(shape)

        dtype = np.dtype(dtype)
        assert dtype.char in 'fdgFDG', f'{dtype} is not a floating point type'

        if padding is not False:
            assert len(padding) == len(shape)
            for ax in axes:
                if len(ax) == 1 and padding[ax[0]] > 1.0 + 1e-6:
                    before = float(shape[ax[0]])
                    shape[ax[0]] = int(np.floor(shape[ax[0]] * padding[ax[0]]))
                    padding[ax[0]] = shape[ax[0]] / before

        self._input_shape = tuple(shape)
        assert len(shape) > 0 and min(shape) > 0

        self.subcomm = self._decompose(comm, shape, axes, grid, kw.pop('slab', False))

        self.collapse = collapse
        if collapse is True:
            # merge neighbouring axes that this rank holds whole, so they are transformed together
            groups = [[]]
            for ax in reversed(axes):
                if all(self.subcomm[axis].Get_size() == 1 for axis in ax):
                    for axis in reversed(ax):
                        groups[0].insert(0, axis)
                else:
                    groups.insert(0, ax)
            axes = groups
        self.axes = tuple(map(tuple, axes))

        self.xfftn = []
        self.transfer = []
        self.pencil = [None, None]

        axes = self.axes[-1]
        pencil = CuPyPencil(self.subcomm, shape, axes[-1])
        xfftn = CuPyFFT(pencil.subshape, axes, dtype, padding, transforms=transforms, **kw)
        self.xfftn.append(xfftn)
        self.pencil[0] = pencilA = pencil

        if not shape[axes[-1]] == xfftn.forward.output_array.shape[axes[-1]]:
            dtype = xfftn.forward.output_array.dtype
            shape[axes[-1]] = xfftn.forward.output_array.shape[axes[-1]]
            pencilA = CuPyPencil(self.subcomm, shape, axes[-1])

        for axes in reversed(self.axes[:-1]):
            pencilB = pencilA.pencil(axes[-1])
            transAB = pencilA.transfer(pencilB, dtype)
            xfftn = CuPyFFT(pencilB.subshape, axes, dtype, padding, transforms=transforms, **kw)
            self.xfftn.append(xfftn)
            self.transfer.append(transAB)
            pencilA = pencilB
            if not shape[axes[-1]] == xfftn.forward.output_array.shape[axes[-1]]:
                dtype = xfftn.forward.output_array.dtype
                shape[axes[-1]] = xfftn.forward.output_array.shape[axes[-1]]
                pencilA = CuPyPencil(pencilB.subcomm, shape, axes[-1])

        self.pencil[1] = pencilA
        self._output_shape = tuple(shape)

        self.forward = Transform([o.forward for o in self.xfftn], [o.forward for o in self.transfer], self.pencil)
        self.backward = Transform(
            [o.backward for o in self.xfftn[::-1]],
            [o.backward for o in self.transfer[::-1]],
            self.pencil[::-1],
        )

    @staticmethod
    def _normalise_axes(axes, shape):
        """Turn whatever was asked for into a list of tuples of non-negative axes."""
        if axes is None:
            axes = list(range(len(shape)))
        else:
            axes = list(axes) if not isinstance(axes, int) else [axes]

        for i, ax in enumerate(axes):
            if isinstance(ax, (int, np.integer)):
                axes[i] = (ax + len(shape) if ax < 0 else ax,)
            else:
                assert isinstance(ax, (tuple, list))
                axes[i] = [a + len(shape) if a < 0 else a for a in ax]
            assert min(axes[i]) >= 0
            assert max(axes[i]) < len(shape)
            assert 0 < len(axes[i]) <= len(shape)
            assert sorted(axes[i]) == sorted(set(axes[i])), 'an axis was asked for twice'
        return axes

    @staticmethod
    def _decompose(comm, shape, axes, grid, slab):
        """Decide which axes are split over which ranks."""
        if grid is not None:
            assert not isinstance(comm, Subcomm) and slab is False
            dims = list(grid) + [1] * (len(shape) - len(grid))
            return Subcomm(comm, dims)

        if isinstance(comm, Subcomm):
            assert slab is False
            assert len(comm) == len(shape)
            assert all(comm[ax].Get_size() == 1 for ax in axes[-1]), 'the first axes to transform must be whole'
            return comm

        if slab is False or slab is None:
            # split everything except the axes transformed first, which have to be local
            dims = [0] * len(shape)
            for ax in axes[-1]:
                dims[ax] = 1
        else:
            axis = (axes[-1][-1] + 1) % len(shape) if slab is True else slab % len(shape)
            dims = [1] * len(shape)
            dims[axis] = comm.Get_size()
        return Subcomm(comm, dims)
