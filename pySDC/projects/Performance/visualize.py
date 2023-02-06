import glob
import json
import numpy as np
import pySDC.helpers.plot_helper as plt_helper


def joint_plots(list_of_result_paths):
    # get result data from JUBE tables
    results = []
    for result_path in list_of_result_paths:
        results.append(np.genfromtxt(result_path, names=True, skip_header=1, delimiter='|', dtype=float, comments='--'))

    # fill arrays with data
    ncores = np.concatenate(results)['ntasks'] * np.concatenate(results)['nnodes']
    timings_space = results[0]['timing_pat']
    timings_spacetime = results[1]['timing_pat']
    ideal = [timings_space[0] / (c / ncores[0]) for c in np.unique(ncores)]

    # setup and fill plots
    plt_helper.setup_mpl()
    plt_helper.newfig(textwidth=238.96, scale=1.0)

    plt_helper.plt.loglog(np.unique(ncores), ideal, 'k--', label='ideal')
    plt_helper.plt.loglog(
        ncores[0 : len(results[0])],
        timings_space,
        lw=1,
        ls='-',
        color='b',
        marker='o',
        markersize=4,
        markeredgecolor='k',
        label='parallel-in-space',
    )
    plt_helper.plt.loglog(
        ncores[-len(results[1]) :],
        timings_spacetime,
        lw=1,
        ls='-',
        color='r',
        marker='d',
        markersize=4,
        markeredgecolor='k',
        label='parallel-in-space-time',
    )

    plt_helper.plt.grid()
    plt_helper.plt.legend(loc=3, ncol=1)
    plt_helper.plt.xlabel('Number of cores')
    plt_helper.plt.ylabel('Time [s]')

    # save plot, beautify
    fname = 'data/scaling'
    plt_helper.savefig(fname)


def plot_data(name=''):
    """
    Visualization using numpy arrays (written via MPI I/O) and json description

    Produces one png file per time-step, combine as movie via e.g.
      > ffmpeg -i data/name_%08d.png name.mp4

    Args:
        name (str): name of the simulation (expects data to be in data path)
    """

    # get data and json files
    json_files = sorted(glob.glob(f'./data/{name}_*.json'))
    data_files = sorted(glob.glob(f'./data/{name}_*.dat'))

    # setup plotting
    plt_helper.setup_mpl()

    for json_file, data_file in zip(json_files, data_files):
        with open(json_file, 'r') as fp:
            obj = json.load(fp)

        index = json_file.split('_')[1].split('.')[0]
        print(f'Working on step {index}...')

        # get data and format
        array = np.fromfile(data_file, dtype=obj['datatype'])
        array = array.reshape(obj['shape'], order='C')

        # plot
        plt_helper.newfig(textwidth=238.96, scale=1.0)
        plt_helper.plt.imshow(array, vmin=0, vmax=1, extent=[-2, 2, -2, 2], origin='lower')

        plt_helper.plt.yticks(range(-2, 3))
        cbar = plt_helper.plt.colorbar()
        cbar.set_label('concentration')
        # plt_helper.plt.title(f"Time: {obj['time']:6.4f}")

        # save plot, beautify
        fname = f'data/{name}_{index}'
        plt_helper.savefig(fname, save_pgf=False, save_png=False)


if __name__ == '__main__':
    list_of_result_paths = [
        'data/bench_run_SPxTS/000004/result/result.dat',
        'data/bench_run_SPxTP/000002/result/result.dat',
    ]

    # joint_plots(list_of_result_paths)
    plot_data(name='AC-bench-noforce')
