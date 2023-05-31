import mpi4py

mpi4py.rc.recv_mprobe = False
import argparse
from pySDC.projects.Monodomain.run_scripts.run_MonodomainODE import setup_and_run


def list_of_ints(arg):
    arg = arg.replace(' ', '')
    arg = arg.replace('_', '-')
    arg = arg.split(',')
    return list(map(int, arg))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # define sweeper parameters
    parser.add_argument("--integrator", default="IMEXEXP_EXPRK", type=str, help="sweeper name")
    parser.add_argument("--num_nodes", default="6", type=list_of_ints, help="list of ints (as '5,3', i.e. no brakets): number of collocation nodes per level")
    parser.add_argument("--num_sweeps", default="1", type=list_of_ints, help="list of ints: number of sweeps per level")
    parser.add_argument("--mass_rhs", default="none", type=str, help="if rhs is already multiplied by mass matrix: none, one (only potential), all (also ionic model states)")
    # set step parameters
    parser.add_argument("--max_iter", default=100, type=int, help="maximal number of iterations")
    # set level parameters
    parser.add_argument("--dt", default=0.1, type=float, help="step size")
    parser.add_argument("--restol", default=5e-8, type=float, help="residual tolerance")
    # problem args
    parser.add_argument("--space_disc", default="FD", type=str, help="space discretization method: FEM, FD")
    parser.add_argument("--domain_name", default="cube_2D", type=str, help="cuboid_2D, cuboid_3D, truncated_ellipsoid,...")
    parser.add_argument("--pre_refinements", default="-1", type=list_of_ints, help="list of ints: loads a mesh which has already been pre-refined pre_refinements times.")
    parser.add_argument("--order", default="4", type=list_of_ints, help="list of ints: order of FEM or FD discretization")
    parser.add_argument("--lin_solv_max_iter", default=1000, type=int, help="maximal number of iterations in iterative linear solver")
    parser.add_argument("--lin_solv_rtol", default=1e-8, type=float, help="residual tolerance in iterative linear solver")
    parser.add_argument("--ionic_model_name", default="TTP", type=str, help="ionic_model: HH, CRN, TTP")
    parser.add_argument("--read_init_val", default=False, action=argparse.BooleanOptionalAction, help="read the initial value from file")
    parser.add_argument("--init_time", default=0.0, type=float, help="duration of stimulus. -1 means default")
    parser.add_argument("--enable_output", default=False, action=argparse.BooleanOptionalAction, help="activate or deactivate xdmf output: True or False")
    parser.add_argument("--write_as_reference_solution", default=False, action=argparse.BooleanOptionalAction, help="write as reference solution: True or False")
    parser.add_argument("--end_time", default=1.0, type=float, help="end time. If negative, a default one is used")
    parser.add_argument("--output_file_name", default="monodomain", type=str, help="output file name")
    parser.add_argument("--ref_sol", default="ref_sol", type=str, help="reference solution file name")
    parser.add_argument("--output_root", default="results_tmp/", type=str, help="output root folder")
    parser.add_argument("--mass_lumping", default=True, action=argparse.BooleanOptionalAction, help="with or without mass lumping")
    # controller args
    parser.add_argument("--truly_time_parallel", default=True, action=argparse.BooleanOptionalAction, help="truly time parallel or emulated")
    parser.add_argument("--n_time_ranks", default=1, type=int, help="number of time ranks")
    parser.add_argument("--print_stats", default=True, action=argparse.BooleanOptionalAction, help="print stats or not")

    args = parser.parse_args()

    err, rel_err = setup_and_run(
        args.integrator,
        args.num_nodes,
        args.num_sweeps,
        args.max_iter,
        args.space_disc,
        args.dt,
        args.restol,
        args.domain_name,
        args.pre_refinements,
        args.order,
        args.mass_lumping,
        args.mass_rhs,
        args.lin_solv_max_iter,
        args.lin_solv_rtol,
        args.ionic_model_name,
        args.read_init_val,
        args.init_time,
        args.enable_output,
        args.write_as_reference_solution,
        args.output_root,
        args.output_file_name,
        args.ref_sol,
        args.end_time,
        args.truly_time_parallel,
        args.n_time_ranks,
        args.print_stats,
    )


if __name__ == "__main__":
    main()