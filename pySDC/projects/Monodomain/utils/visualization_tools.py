import matplotlib

matplotlib.use("Agg")

from pySDC.helpers.stats_helper import filter_stats
from collections import namedtuple

import numpy as np

from matplotlib import rc
import matplotlib.pyplot as plt


def sort_stats(stats, sortby_list, comm=None):
    """
    Helper function to transform stats dictionary to sorted list of tuples.
    This is very similar to the sort_stats function already provided in pySDC
    but here we can sort with respect to more than one key in stats, hence
    we sort by first key and for items where first key is the same we sort
    by second key and so on.

    Args:
        stats (dict): dictionary of statistics
        sortby_list (str): list of strings to specify which keys to use for sorting
        comm (mpi4py.MPI.Intracomm): Communicator (or None if not applicable)

    Returns:
        list: list of tuples containing the sortby_list items and the value
    """

    result = []
    for k, v in stats.items():
        # convert string to attribute and append key + value to result as tuple
        items = [getattr(k, sortby) for sortby in sortby_list]
        result.append((items, v))

    if comm is not None:
        # gather the results across all ranks and the flatten the list
        result = [item for sub_result in comm.allgather(result) for item in sub_result]

    # sort by first element of the tuple (which is the sortby key) and return
    sorted_data = sorted(result, key=lambda tup: tup[0])

    return sorted_data


def tuple_to_stats(tup_list, list_str):
    entry = namedtuple("Entry", list_str)
    stats = dict()
    for t in tup_list:
        stats[entry(*t[0])] = t[1]

    return stats


# noinspection PyShadowingBuiltins
def show_residual_across_simulation(stats, fname="residuals.png", comm=None, tend=None):
    """
    Helper routine to visualize the residuals across the simulation (one block of PFASST)

    Args:
        stats (dict): statistics object from a PFASST run
        fname (str): filename
    """

    # get residuals of the run
    extract_stats = filter_stats(stats, type="residual_post_iteration")

    sortby_list = ["time", "iter", "process"]  # process is not used to sort since is similar to time, but I want to keep it in the result
    sorted_stats = sort_stats(extract_stats, sortby_list, comm)  # not really for sorting but more for comminucating across ranks
    dt = sorted_stats[0][0][0]
    gathered_stats = tuple_to_stats(sorted_stats, sortby_list)
    del extract_stats, sortby_list, sorted_stats

    # find boundaries for x-,y- and c-axis as well as arrays
    maxprocs = 0
    maxiter = 0
    minres = 0
    maxres = -99
    for k, v in gathered_stats.items():
        maxprocs = max(maxprocs, k.process)
        maxiter = max(maxiter, k.iter)
        minres = min(minres, np.log10(v))
        maxres = max(maxres, np.log10(v))

    times = np.array([])
    procs = np.array([])
    for k, v in gathered_stats.items():
        if not np.any(np.isclose(times - k.time, np.zeros_like(times))):
            times = np.append(times, k.time)
            procs = np.append(procs, k.process)
    if times.size > 1:
        dt = times[1]
    else:
        dt = tend
    steps = list(range(times.shape[0]))
    n_steps = steps[-1] + 1

    # grep residuals and put into array
    log_residual = np.zeros((maxiter, n_steps))
    residual = np.zeros((maxiter, n_steps))
    log_residual[:] = -99
    residual[:] = 0.0
    for k, v in gathered_stats.items():
        step = np.round(k.time / dt).astype(int)
        iter = k.iter
        if iter != -1:
            log_residual[iter - 1, step] = np.log10(v)
            residual[iter - 1, step] = v

    # Set up plotting stuff and fonts
    rc("font", **{"size": 30})
    rc("legend", fontsize="small")
    rc("xtick", labelsize="small")
    rc("ytick", labelsize="small")

    # create plot and save
    fig, ax = plt.subplots(figsize=(15, 10))

    cmap = plt.get_cmap("Reds")
    plt.pcolor(log_residual.T, cmap=cmap, vmin=minres, vmax=maxres)

    cax = plt.colorbar()
    cax.set_label("log10(residual)")

    ax.set_xlabel("SDC Tteration")
    ax.set_ylabel("Time")

    plt.savefig(fname, transparent=True, bbox_inches="tight")
