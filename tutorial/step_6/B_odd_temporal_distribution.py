import numpy as np
import os

from pySDC.implementations.problem_classes.HeatEquation_1D_FD import heat1d
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.sweeper_classes.generic_LU import generic_LU
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI
from pySDC.implementations.controller_classes.allinclusive_multigrid_nonMPI import allinclusive_multigrid_nonMPI

from pySDC.plugins.stats_helper import filter_stats, sort_stats
from tutorial.step_6.A_classic_vs_multigrid_controller import main as main_A


def main():
    """
    A simple test program to do PFASST runs for the heat equation
    """
    main_A(num_proc_list=[3,5,7,9], fname='step_6_B_out.txt')


if __name__ == "__main__":
    main()
