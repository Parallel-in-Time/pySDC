from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_forced
from pySDC.implementations.problem_classes.AllenCahn_MPIFFT import (
    allencahn_imex_timeforcing,
)

from pySDC.projects.compression.compressed_mesh import (
    compressed_mesh,
    imex_mesh_compressed,
)


class heat_ND_compressed(heatNd_forced):
    dtype_f = imex_mesh_compressed
    dtype_u = compressed_mesh


class AllenCahn_MPIFFT_Compressed(allencahn_imex_timeforcing):
    dtype_f = imex_mesh_compressed
    dtype_u = compressed_mesh
