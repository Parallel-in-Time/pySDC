import pySDC.helpers.plot_helper as plt_helper

def beautify_plot(nprocs, fname):
    plt_helper.plt.grid()
    plt_helper.plt.legend(loc=0)
    plt_helper.plt.xlabel('Number of parallel steps')
    plt_helper.plt.ylabel('Theoretical speedup')

    plt_helper.plt.xlim(0.9 * nprocs[0], 1.1 * nprocs[-1])
    plt_helper.plt.ylim(0.25, 5.25)

    plt_helper.plt.xticks(nprocs, nprocs)
    plt_helper.plt.minorticks_off()

    # save plot, beautify
    plt_helper.savefig(fname)

def plot_data():

    nprocs = [1, 2, 4, 8, 16]

    niter_fd = [2.8125, 3.875, 5.5, 7.5, 9.0]
    niter_fft_overres = [2.8125, 3.875, 5.5, 7.5, 9.0]
    niter_fft = [6.3125, 8.375, 11.25, 15.5, 24.0]
    niter_fft_slight = [2.8125, 3.875, 5.5, 7.5, 9.0]

    speedup_fd = [p / (i / niter_fd[0]) for p, i in zip(nprocs, niter_fd)]
    speedup_fft_overres = [p / (i / niter_fft_overres[0]) for p, i in zip(nprocs, niter_fft_overres)]
    speedup_fft = [p / (i / niter_fft_overres[0]) for p, i in zip(nprocs, niter_fft)]
    speedup_fft_slight = [p / (i / niter_fft_overres[0]) for p, i in zip(nprocs, niter_fft_slight)]

    plt_helper.setup_mpl()
    plt_helper.newfig(textwidth=238.96, scale=1.0)
    plt_helper.plt.semilogx(nprocs, speedup_fd, color='b', marker='o', markersize=6, label='FD, Nx=128')
    beautify_plot(nprocs, 'fool_speedup_fd')

    plt_helper.setup_mpl()
    plt_helper.newfig(textwidth=238.96, scale=1.0)
    plt_helper.plt.semilogx(nprocs, speedup_fd, color='b', marker='o', markersize=6, label='FD, Nx=128')
    plt_helper.plt.semilogx(nprocs, speedup_fft_overres, color='orange', marker='x', markersize=6, label='Spectral, Nx=128')
    beautify_plot(nprocs, 'fool_speedup_fft_overres')

    plt_helper.setup_mpl()
    plt_helper.newfig(textwidth=238.96, scale=1.0)
    plt_helper.plt.semilogx(nprocs, speedup_fd, color='b', marker='o', markersize=6, label='FD, Nx=128')
    plt_helper.plt.semilogx(nprocs, speedup_fft_overres, color='orange', marker='x', markersize=6, label='Spectral, Nx=128')
    plt_helper.plt.semilogx(nprocs, speedup_fft, color='r', marker='v', markersize=6, label='Spectral, Nx=8')
    beautify_plot(nprocs, 'fool_speedup_fft_minimal')

    plt_helper.setup_mpl()
    plt_helper.newfig(textwidth=238.96, scale=1.0)
    plt_helper.plt.semilogx(nprocs, speedup_fd, color='b', marker='o', markersize=6, label='FD, Nx=128')
    plt_helper.plt.semilogx(nprocs, speedup_fft_overres, color='orange', marker='x', markersize=6, label='Spectral, Nx=128')
    plt_helper.plt.semilogx(nprocs, speedup_fft, color='r', marker='v', markersize=6, label='Spectral, Nx=8')
    plt_helper.plt.semilogx(nprocs, speedup_fft_slight, color='g', marker='d', markersize=6, label='Spectral, Nx=16')
    beautify_plot(nprocs, 'fool_speedup_fft_slight')



    # plt_helper.plt.grid()
    # plt_helper.plt.legend(loc=0)
    # plt_helper.plt.xlabel('Number of parallel steps')
    # plt_helper.plt.ylabel('Theoretical speedup')
    #
    # plt_helper.plt.xlim(0.9*nprocs[0], 1.1*nprocs[-1])
    # plt_helper.plt.ylim(0.75, 5.25)
    #
    # plt_helper.plt.xticks(nprocs, nprocs)
    # plt_helper.plt.minorticks_off()
    #
    # # save plot, beautify
    # fname = 'fool_speedup_fd'
    # plt_helper.savefig(fname)

    # plt_helper.plt.semilogx(nprocs, speedup_fft, color='g', marker='o', label='FFT')
    #
    # fname = 'fool_speedup_fft'
    # plt_helper.savefig(fname)



if __name__ == '__main__':

    plot_data()