from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_forced
from pySDC.playgrounds.compression.compressed_mesh import (
    compressed_mesh,
    imex_mesh_compressed,
)


class heat_ND_compressed(heatNd_forced):
    dtype_f = imex_mesh_compressed
    dtype_u = compressed_mesh
