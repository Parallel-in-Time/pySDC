import os

import matplotlib.colors as colors
import numpy as np

import pySDC.helpers.plot_helper as plt_helper


def is_number(s):
    """
    Helper function to detect numbers

    Args:
        s: a string

    Returns:
        bool: True if s is a number
    """
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def join_timings(file=None, result=None, cwd=''):
    """
    Helper function to read in JUBE result tables and convert/join them into a single dictionary

    Args:
        file: current fils containing a JUBE result table
        result: dictionary (empty or not)
        cwd (str): current working directory

    Returns:
        dict: result dictionary for further usage
    """
    with open(cwd + file) as f:
        lines = f.readlines()

    for line in lines:
        line_split = line.replace('\n', '').replace(' ', '').split('|')
        if is_number(line_split[0]):
            ntime = int(int(line_split[0]) * int(line_split[1]) / int(line_split[2]))
            nspace = int(line_split[2])
            timing = float(line_split[3])
            result[(nspace, ntime)] = timing

    return result


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Helper function to crop a colormap

    Args:
        cmap: colormap
        minval: minimum value
        maxval: maximum value
        n: stepsize

    Returns:
        cropped colormap
    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def visualize_matrix(result=None):
    """
    Visualizes runtimes in a matrix (cores in space vs. cores in time)

    Args:
        result: dictionary containing the runtimes
    """
    process_list = [1, 2, 4, 6, 12, 24]
    dim = len(process_list)
    mat = np.zeros((dim, dim))
    tmin = 1E03
    tmax = 0
    for key, item in result.items():
        mat[process_list.index(key[0]), process_list.index(key[1])] = item
        tmin = min(tmin, item)
        tmax = max(tmax, item)

    plt_helper.setup_mpl()
    plt_helper.newfig(textwidth=120, scale=1.5)
    cmap = plt_helper.plt.get_cmap('RdYlGn_r')
    new_cmap = truncate_colormap(cmap, 0.1, 0.9)
    plt_helper.plt.imshow(mat, origin='lower', norm=colors.LogNorm(vmin=tmin, vmax=tmax), cmap=new_cmap, aspect='auto')

    for key, item in result.items():
        timing = "{:3.1f}".format(item)
        plt_helper.plt.annotate(timing, xy=(process_list.index(key[0]), process_list.index(key[1])), size='x-small',
                                ha='center', va='center')

    plt_helper.plt.xticks(range(dim), process_list)
    plt_helper.plt.yticks(range(dim), process_list)
    plt_helper.plt.xlabel('Cores in space')
    plt_helper.plt.ylabel('Cores in time')

    fname = 'data/runtimes_matrix_heat'
    plt_helper.savefig(fname)

    assert os.path.isfile(fname + '.pdf'), 'ERROR: plotting did not create PDF file'
    assert os.path.isfile(fname + '.pgf'), 'ERROR: plotting did not create PGF file'
    assert os.path.isfile(fname + '.png'), 'ERROR: plotting did not create PNG file'


def visualize_speedup(result=None):
    """
    Visualizes runtimes of two different runs (MLSDC vs. PFASST)

    Args:
        result: dictionary containing the runtimes
    """
    process_list_MLSDC = [1, 2, 4, 6, 12, 24]
    process_list_PFASST = [24, 48, 96, 144, 288, 576]

    timing_MLSDC = np.zeros(len(process_list_MLSDC))
    timing_PFASST = np.zeros((len(process_list_PFASST)))
    for key, item in result.items():
        if key[0] * key[1] in process_list_MLSDC:
            timing_MLSDC[process_list_MLSDC.index(key[0] * key[1])] = item
        if key[0] * key[1] in process_list_PFASST:
            timing_PFASST[process_list_PFASST.index(key[0] * key[1])] = item

    plt_helper.setup_mpl()
    plt_helper.newfig(textwidth=120, scale=1.5)

    process_list_all = process_list_MLSDC + process_list_PFASST
    ideal = [timing_MLSDC[0] / nproc for nproc in process_list_all]
    plt_helper.plt.loglog(process_list_all, ideal, 'k--', label='ideal')
    plt_helper.plt.loglog(process_list_MLSDC, timing_MLSDC, 'bo-', label='MLSDC')
    plt_helper.plt.loglog(process_list_PFASST, timing_PFASST, 'rs-', label='PFASST')

    plt_helper.plt.xlim(process_list_all[0] / 2, process_list_all[-1] * 2)
    plt_helper.plt.ylim(ideal[-1] / 2, ideal[0] * 2)
    plt_helper.plt.xlabel('Number of cores')
    plt_helper.plt.ylabel('Runtime (sec.)')

    plt_helper.plt.legend()
    plt_helper.plt.grid()

    fname = 'data/speedup_heat'
    plt_helper.savefig(fname)
    assert os.path.isfile(fname + '.pdf'), 'ERROR: plotting did not create PDF file'
    assert os.path.isfile(fname + '.pgf'), 'ERROR: plotting did not create PGF file'
    assert os.path.isfile(fname + '.png'), 'ERROR: plotting did not create PNG file'


def main(cwd=''):
    """
    Main routine to call them all

    Args:
        cwd (str): current working directory

    """
    result = {}
    files = ['data/result_MLSDC.dat', 'data/result_PFASST_2.dat', 'data/result_PFASST_4.dat',
             'data/result_PFASST_6.dat', 'data/result_PFASST_12.dat', 'data/result_PFASST_24.dat']
    for file in files:
        result = join_timings(file=file, result=result, cwd=cwd)
    visualize_matrix(result=result)

    result = {}
    files = ['data/result_MLSDC.dat', 'data/result_PFASST_multinode_24.dat']
    for file in files:
        result = join_timings(file=file, result=result, cwd=cwd)
    visualize_speedup(result=result)


if __name__ == "__main__":
    main()
