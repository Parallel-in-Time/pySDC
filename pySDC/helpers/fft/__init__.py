"""
pySDC's own distributed FFT for GPUs.

``mpi4py-fft`` distributes an array over a pencil decomposition and transforms it axis by axis,
redistributing in between. Its released version holds the data in NumPy arrays and moves it with
``MPI_Alltoallw``, so it cannot drive a GPU; the CuPy support pySDC needs has sat in an unmerged
pull request since February 2024, and pySDC was running on a personal fork of it.

What is backend-agnostic in ``mpi4py-fft`` is reused rather than reimplemented. ``Subcomm`` and
``Pencil`` are index arithmetic with no array in them, and ``FFTBase`` carries the padding and
truncation that dealiasing needs, which is subtle and identical in the fork. What this package adds
is the three things that genuinely need a GPU: an array in device memory, a transpose that moves it
with NCCL rather than MPI, and serial transforms from ``cupyx.scipy.fft``.

Nothing here is imported unless a run asks for a GPU, so a CPU-only installation never needs
``cupy``. The CPU path keeps using released ``mpi4py-fft``, which also makes it the oracle these
are tested against: the same run on CPU and GPU has to give the same numbers.
"""


def __getattr__(name):
    """Import the GPU pieces only when something asks for them."""
    if name in ('DistArrayCuPy', 'PFFT_GPU'):
        from importlib import import_module

        module = '_distarray' if name == 'DistArrayCuPy' else '_pfft'
        return getattr(import_module(f'pySDC.helpers.fft.{module}'), name)
    raise AttributeError(name)


def newDistArray(pfft, forward_output=True, val=0, rank=0, view=False):
    """An empty distributed array of the shape and type ``pfft`` transforms.

    The same role as :func:`mpi4py_fft.distarray.newDistArray`, but dispatching on where the
    transform keeps its data rather than always building a NumPy-backed array.
    """
    global_shape = pfft.global_shape(forward_output)
    p0 = pfft.pencil[forward_output]
    dtype = pfft.forward.output_array.dtype if forward_output else pfft.forward.input_array.dtype
    global_shape = (len(global_shape),) * rank + global_shape

    if getattr(pfft, 'on_GPU', False):
        from pySDC.helpers.fft._distarray import DistArrayCuPy as cls
    else:
        from mpi4py_fft.distarray import DistArray as cls

    array = cls(global_shape, subcomm=p0.subcomm, val=val, dtype=dtype, alignment=p0.axis, rank=rank)
    return array.v if view else array
