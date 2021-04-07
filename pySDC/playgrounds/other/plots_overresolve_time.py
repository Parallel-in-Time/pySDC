import pySDC.helpers.plot_helper as plt_helper

def beautify_plot(nprocs, fname):
    plt_helper.plt.grid()
    plt_helper.plt.legend(loc=2)
    plt_helper.plt.xlabel('Number of parallel steps')
    plt_helper.plt.ylabel('Theoretical speedup')

    plt_helper.plt.xlim(0.9 * nprocs[0], 1.1 * nprocs[-1])
    plt_helper.plt.ylim(0.25, 6.5)

    plt_helper.plt.xticks(nprocs, nprocs)
    plt_helper.plt.minorticks_off()

    # save plot, beautify
    plt_helper.savefig(fname)

def plot_data():

    nprocs = [1, 2, 4, 8, 16, 32]
    niter_overres = [1, 2, 3, 3, 3, 4]
    alpha_overres = 1.0 / 32.0
    speedup_overres = [p / (p * alpha_overres + k * (1 + alpha_overres)) for p, k in zip(nprocs, niter_overres)]

    plt_helper.setup_mpl()
    plt_helper.newfig(textwidth=238.96, scale=1.0)
    plt_helper.plt.semilogx(nprocs, speedup_overres, color='g', marker='o', markersize=6, label=r'$N_\mathcal{F}=1024, \alpha=\frac{1}{32}$')
    beautify_plot(nprocs, 'fool_speedup_overres_time')

    niter_wellres_1 = [1, 2, 4, 8, 16, 32]
    alpha_wellres_1 = 1.0 / 32.0
    speedup_wellres_1 = [p / (p * alpha_wellres_1 + k * (1 + alpha_wellres_1)) for p, k in zip(nprocs, niter_wellres_1)]

    niter_wellres_2 = [1, 2, 4, 5, 6, 8]
    alpha_wellres_2 = 1.0 / 4.0
    speedup_wellres_2 = [p / (p * alpha_wellres_2 + k * (1 + alpha_wellres_2)) for p, k in zip(nprocs, niter_wellres_2)]

    # niter_wellres_3 = [1, 2, 3, 3, 4, 5]
    # alpha_wellres_3 = 1.0 / 2.0
    # speedup_wellres_3 = [p / (p * alpha_wellres_3 + k * (1 + alpha_wellres_3)) for p, k in zip(nprocs, niter_wellres_3)]

    plt_helper.setup_mpl()
    plt_helper.newfig(textwidth=238.96, scale=1.0)
    plt_helper.plt.semilogx(nprocs, speedup_wellres_1, color='r', marker='d', markersize=6, label=r'$N_\mathcal{F}=32, \alpha=\frac{1}{32}$')
    plt_helper.plt.semilogx(nprocs, speedup_wellres_2, color='b', marker='s', markersize=6, label=r'$N_\mathcal{F}=32, \alpha=\frac{1}{4}$')
    # plt_helper.plt.semilogx(nprocs, speedup_wellres_3, color='c', marker='v', markersize=6, label=r'$Nt_f=32, \alpha=\frac{1}{2}$')
    beautify_plot(nprocs, 'fool_speedup_wellres_time')


if __name__ == '__main__':

    plot_data()