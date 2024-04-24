import argparse
from pySDC.projects.Monodomain.run_scripts.run_MonodomainODE import setup_and_run


def list_of_ints(arg):
    arg = arg.replace(' ', '')
    arg = arg.replace('_', '-')
    arg = arg.split(',')
    return list(map(int, arg))


# This is to run the MonodomainODE example from the command line
# Pretty much all the parameters can be defined from the command line

# For the refinements, it is possible to set negative values, which yield a mesh coarser than the baseline.
# To do so in the command line use an underscore _ insteaf of a minus sign -.
# For example, to solve a 3 level example with meshes refinements 1, 0 and -1, use the option --refinements 1,0,_1


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # define sweeper parameters
    parser.add_argument("--integrator", default="IMEXEXP_EXPRK", type=str, help="sweeper name")
    parser.add_argument(
        "--num_nodes",
        default="4",
        type=list_of_ints,
        help="list of ints (as '5,3', i.e. no brackets): number of collocation nodes per level",
    )
    parser.add_argument("--num_sweeps", default="1", type=list_of_ints, help="list of ints: number of sweeps per level")
    parser.add_argument(
        "--skip_res",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="compute residual only when really needed",
    )
    # set step parameters
    parser.add_argument("--max_iter", default=100, type=int, help="maximal number of iterations")
    # set level parameters
    parser.add_argument("--dt", default=0.05, type=float, help="step size")
    parser.add_argument("--restol", default=5e-8, type=float, help="residual tolerance")
    # problem args
    parser.add_argument(
        "--domain_name", default="cuboid_2D_small", type=str, help="cuboid_2D, cuboid_3D, truncated_ellipsoid,..."
    )
    parser.add_argument(
        "--refinements",
        default="0",
        type=list_of_ints,
        help="list of ints: number of refinements per level, with respect to a baseline mesh (negative values yield coarser meshes). For negative values use _ instead of -.",
    )
    parser.add_argument(
        "--order", default="4", type=list_of_ints, help="list of ints: order of FEM or FD discretization"
    )
    parser.add_argument("--ionic_model_name", default="TTP", type=str, help="ionic_model: HH, CRN, TTP")
    parser.add_argument(
        "--read_init_val", default=False, action=argparse.BooleanOptionalAction, help="read the initial value from file"
    )
    parser.add_argument("--init_time", default=0.0, type=float, help="duration of stimulus. -1 means default")
    parser.add_argument(
        "--enable_output",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="activate or deactivate xdmf output: True or False",
    )
    parser.add_argument(
        "--write_as_reference_solution",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="write as reference solution: True or False",
    )
    parser.add_argument(
        "--write_all_variables",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="when write_as_reference_solution=True, write write all variables (True) or only potential V (False)",
    )
    parser.add_argument("--end_time", default=1.0, type=float, help="end time. If negative, a default one is used")
    parser.add_argument("--output_file_name", default="monodomain", type=str, help="output file name")
    parser.add_argument("--ref_sol", default="ref_sol", type=str, help="reference solution file name")
    parser.add_argument("--output_root", default="results_tmp/", type=str, help="output root folder")
    parser.add_argument(
        "--finter",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="in prolong, re-evaluate f (false) or interpolate (true)",
    )
    # controller args
    parser.add_argument(
        "--truly_time_parallel",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="truly time parallel or emulated",
    )
    parser.add_argument("--n_time_ranks", default=1, type=int, help="number of time ranks")

    parser.add_argument(
        "--write_database",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="save some simulation results in a database",
    )

    args = parser.parse_args()

    error_L2, rel_error_L2, avg_niters, times, niters, residuals = setup_and_run(
        args.integrator,
        args.num_nodes,
        args.skip_res,
        args.num_sweeps,
        args.max_iter,
        args.dt,
        args.restol,
        args.domain_name,
        args.refinements,
        args.order,
        args.ionic_model_name,
        args.read_init_val,
        args.init_time,
        args.enable_output,
        args.write_as_reference_solution,
        args.write_all_variables,
        args.output_root,
        args.output_file_name,
        args.ref_sol,
        args.end_time,
        args.truly_time_parallel,
        args.n_time_ranks,
        args.finter,
        args.write_database,
    )


if __name__ == "__main__":
    main()
