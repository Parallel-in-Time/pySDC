from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_forced
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.helpers.stats_helper import get_sorted
from mpi4py import MPI
from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun
from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.projects.compression.compressed_problems import heat_ND_compressed
from pySDC.projects.compression.log_datatype_creations import LogDatatypeCreations
from pySDC.projects.compression.log_compression_ratio import LogCompressionRatio, LogCacheSize
from pySDC.projects.compression.log_num_registered_var import LogRegisteredVar, LogActiveRegisteredVar
from pySDC.projects.compression.log_time_metrics import LogTimeCompression, LogTimeDecompression
from pySDC.projects.compression.log_num_comp_decomp_calls import LogCompDecompCalls, LogCompCalls, LogDecompCalls
from pySDC.projects.compression.log_clear_stats import LogClearStats
from pySDC.projects.compression.log_cache_invalidates import LogCacheInvalidates
from pySDC.projects.compression.log_cache_history import LogCacheHistory
from pySDC.projects.compression.compression_convergence_controller import (
    Compression_Conv_Controller,
)


def run_heat(residual_tolerance=1e-4, errBound=1e-8, resolution=64, Tend=1):
    # setup communicator
    # comm = MPI.COMM_WORLD if comm is None else comm

    # initialize problem parameters
    problem_params = {}
    problem_params["nu"] = 1.0
    problem_params["freq"] = (4, 4, 4)
    problem_params["order"] = 4
    problem_params["lintol"] = 1e-7
    problem_params["liniter"] = 99
    problem_params["solver_type"] = "CG"
    problem_params["nvars"] = (
        resolution,
        resolution,
        resolution,
    )  # Have to be the same, Nx = Ny = Nz
    problem_params["bc"] = "periodic"

    convergence_controllers = {}
    convergence_controllers[Compression_Conv_Controller] = {"errBound": errBound}

    # initialize level parameters
    level_params = {}
    level_params["restol"] = residual_tolerance
    level_params["dt"] = 1e-01
    level_params["nsweeps"] = 1

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params["node_type"] = "LEGENDRE"
    sweeper_params["quad_type"] = "RADAU-RIGHT"
    sweeper_params["QI"] = ["IE"]
    sweeper_params["QE"] = ["PIC"]
    sweeper_params["num_nodes"] = 3
    sweeper_params["initial_guess"] = "spread"

    # initialize step parameters
    step_params = {}
    step_params["maxiter"] = 50  # 50

    # initialize controller parameters
    controller_params = {}
    controller_params["logger_level"] = 15
    controller_params["hook_class"] = [
        LogSolution,
        LogRegisteredVar,
        LogActiveRegisteredVar,
        LogCompressionRatio,
        LogCacheSize,
        LogTimeCompression,
        LogTimeDecompression,
        LogCompCalls,
        LogDecompCalls,
        LogCacheInvalidates,
        LogClearStats
        # LogCacheHistory,
        # LogDatatypeCreations,
    ]

    # fill description dictionary for easy step instantiation
    description = {}
    # description['problem_class'] = heatNd_forced
    description["problem_class"] = heat_ND_compressed
    description["problem_params"] = problem_params
    description["sweeper_class"] = imex_1st_order
    description["sweeper_params"] = sweeper_params
    description["level_params"] = level_params
    description["step_params"] = step_params
    description["convergence_controllers"] = convergence_controllers

    # instantiate controller
    controller = controller_nonMPI(controller_params=controller_params, description=description, num_procs=1)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(0.0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=0.0, Tend=Tend)

    return stats


def main():
    from pySDC.helpers.stats_helper import get_list_of_types, sort_stats, filter_stats
    import pandas as pd

    stats = run_heat(Tend=0.1)
    # error = max([me[1] for me in get_sorted(stats, type="e_global_post_run")])
    # print(get_list_of_types(stats))
    # print("filter_stats", filter_stats(stats, type="u"))
    # print("sort_stats", sort_stats(filter_stats(stats, type="u"), sortby="time"))
    # u = get_sorted(stats, type="num_datatype_creations")
    # print(u)
    # print(error)
    # import matplotlib.pyplot as plt
    u = get_sorted(stats, type="compression_ratio")
    v = get_sorted(stats, type="num_registered_var")
    w = get_sorted(stats, type="num_active_registered_var")
    cs = get_sorted(stats, type="num_arrays_cache")
    y = get_sorted(stats, type="num_comp_calls")
    z = get_sorted(stats, type="num_decomp_calls")
    time_comp_nocache = get_sorted(stats, type="execution_time_comp_nocache")  # No cache
    time_decomp_nocache = get_sorted(stats, type="execution_time_decomp_nocache")
    time_comp = get_sorted(
        stats, type="execution_time_comp"
    )  # Time - not found in cache - compress, eviction maybe, add to cache (put)
    time_decomp = get_sorted(stats, type="execution_time_decomp")
    time_comp_update = get_sorted(stats, type="execution_time_comp_update")  # Found in cache - update or get it
    time_decomp_get = get_sorted(stats, type="decomp_time_get")
    time_comp_eviction = get_sorted(stats, type="comp_time_eviction")  # Eviction time for write back cache
    time_decomp_eviction = get_sorted(stats, type="decomp_time_eviction")
    time_comp_put = get_sorted(stats, type="comp_time_put")  # Only put operation
    time_decomp_put = get_sorted(stats, type="decomp_time_put")
    t_comp = get_sorted(stats, type="total_time_comp")  # Total time
    t_decomp = get_sorted(stats, type="total_time_decomp")
    cache_inv = get_sorted(stats, type="cache_invalidates")
    # cache_hist = get_sorted(stats, type="cache_history")
    iters = get_sorted(stats, type="niter")
    # cache_hits = get_sorted(stats, type="cache_hits")
    # cache_misses = get_sorted(stats, type="cache_misses")
    print("Compression Ratio:")
    print(u)
    print("\nNumber of registered variables:")
    print(v)
    print("\nNumber of active registered variables:")
    print(w)
    print("\nNumber of compression calls:")
    print(y)
    print("\nNumber of decompression calls:")
    print(z)
    print("\nNumber of cache invalidates:")
    print(cache_inv)
    print("Wrapper compression time nocache:\n")
    print(time_comp_nocache)
    print("Wrapper decompression time nocache:\n")
    print(time_decomp_nocache)
    print("Wrapper compression time:\n")
    print(time_comp)
    print("Wrapper decompression time:\n")
    print(time_decomp)
    print("Wrapper compression time update:\n")
    print(time_comp_update)
    print("Wrapper decompression time get:\n")
    print(time_decomp_get)
    print("Wrapper compression time eviction:\n")
    print(time_comp_eviction)
    print("Wrapper decompression time eviction:\n")
    print(time_decomp_eviction)
    print("Wrapper compression time put:\n")
    print(time_comp_put)
    print("Wrapper decompression time put:\n")
    print(time_decomp_put)

    x, comp_calls = zip(*y)
    x, decomp_calls = zip(*z)
    x, time_comp_nocache = zip(*time_comp_nocache)
    x, time_comp = zip(*time_comp)
    x, time_comp_update = zip(*time_comp_update)
    x, time_comp_eviction = zip(*time_comp_eviction)
    x, time_comp_put = zip(*time_comp_put)

    x, time_decomp_nocache = zip(*time_decomp_nocache)
    x, time_decomp = zip(*time_decomp)
    x, time_decomp_get = zip(*time_decomp_get)
    x, time_decomp_eviction = zip(*time_decomp_eviction)
    x, time_decomp_put = zip(*time_decomp_put)
    x, cache_inv = zip(*cache_inv)
    x, iters = zip(*iters)
    x, t_comp = zip(*t_comp)
    x, t_decomp = zip(*t_decomp)
    df = {
        'Time Steps': x,
        'Compression Calls': comp_calls,
        'Decompression Calls': decomp_calls,
        'Compression Time (No Cache)': time_comp_nocache,
        'Compression Time (Cache)': time_comp,
        'Compression Time (Evictions)': time_comp_eviction,
        'Compression Time (Put)': time_comp_put,
        'Compression Time (Update)': time_comp_update,
        'Decompression Time (No Cache)': time_decomp_nocache,
        'Decompression Time (Cache)': time_decomp,
        'Decompression Time (Evictions)': time_decomp_eviction,
        'Decompression Time (Put)': time_decomp_put,
        'Decompression Time (Get)': time_decomp_get,
        'Total Time (compression)': t_comp,
        'Total Time (decompression)': t_decomp,
        'Cache Invalidates': cache_inv,
        'Number of iterations': iters,
    }
    data_frame = pd.DataFrame(df)
    data_frame.to_csv("result_cache_size_he_" + str(cs[0][1]) + "_nolog_blosc.csv", index=False)
    # print("Wrapper compression time:\n")
    # print("Wrapper decompression time:\n")
    # with open("result_cache_size_"+str(cs[0][1])+"_.txt",'w') as fp:
    #     fp.write(' '.join('{} {}'.format(x[0],x[1]) for x in cs))
    #     fp.write('\n')#Number of Compression calls
    #     fp.write(' '.join('{} {}'.format(x[0],x[1]) for x in y))
    #     fp.write('\n')#Number of Decompression calls
    #     fp.write(' '.join('{} {}'.format(x[0],x[1]) for x in z))
    # fp.close()

    # plt.plot([me[0] for me in u], [me[1] for me in u])
    # plt.show()


if __name__ == "__main__":
    main()
