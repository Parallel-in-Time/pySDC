import pySDC.helpers.plot_helper as plt_helper

import numpy as np
import matplotlib.colors as colors

def is_number(s):
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


def join_timings(file=None, result=None):

    with open(file) as f:
        lines = f.readlines()

    for line in lines:
        line_split = line.replace('\n', '').replace(' ','').split('|')
        if is_number(line_split[0]):
            ntime = int(int(line_split[1]) / int(line_split[2]))
            nspace = int(line_split[2])
            timing = float(line_split[3])
            result[(nspace, ntime)] = timing

    return result


def main():
    result = {}
    files = ['data/result_MLSDC.dat', 'data/result_PFASST_2.dat', 'data/result_PFASST_4.dat',
             'data/result_PFASST_6.dat', 'data/result_PFASST_12.dat', 'data/result_PFASST_24.dat']
    for file in files:
        result = join_timings(file=file, result=result)
    visualize(result=result)
    # print(result)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def visualize(result=None):

    process_list = [1, 2, 4, 6, 12, 24]
    dim = len(process_list)
    mat = np.zeros((dim, dim))
    tmin = 1E03
    tmax = 0
    for key, item in result.items():
        mat[process_list.index(key[0]), process_list.index(key[1])] = item
        tmin = min(tmin, item)
        tmax = max(tmax, item)

    # plt_helper.mpl.style.use('classic')
    plt_helper.setup_mpl()
    plt_helper.newfig(textwidth=238.96, scale=1.33)

    cmap = plt_helper.plt.get_cmap('RdYlGn_r')
    new_cmap = truncate_colormap(cmap, 0.1, 0.9)
    plt_helper.plt.imshow(mat, origin='lower', norm=colors.LogNorm(vmin=tmin, vmax=tmax), cmap=new_cmap)

    for key, item in result.items():
        timing = "{:3.1f}".format(item)
        plt_helper.plt.text(process_list.index(key[0]), process_list.index(key[1]), timing, ha='center', va='center')

    plt_helper.plt.xticks(range(dim), process_list)
    plt_helper.plt.yticks(range(dim), process_list)
    plt_helper.plt.xlabel('Cores in space')
    plt_helper.plt.ylabel('Cores in time')

    fname = 'data/runtimes_matrix'
    plt_helper.savefig(fname)
    pass


if __name__ == "__main__":
    main()