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

    nprocs = [1, 2, 4, 8]
    niter_overres = [9, 5, 11, 23]
    alpha_overres = 1.0 / 4.0
    speedup_overres = [p / (p / niter_overres[0] * alpha_overres + k / niter_overres[0] * (1 + alpha_overres)) for p, k in zip(nprocs, niter_overres)]

    plt_helper.setup_mpl()
    plt_helper.newfig(textwidth=238.96, scale=1.0)
    plt_helper.plt.semilogx(nprocs, speedup_overres, color='orange', marker='o', markersize=6, label=r'$Nx_\mathcal{F}=512, \alpha=\frac{1}{4}$')
    beautify_plot(nprocs, 'fool_speedup_overres_iter')

    niter_wellres_1 = [9, 11, 16, 28]
    alpha_wellres_1 = 1.0 / 4.0
    speedup_wellres_1 = [p / (p / niter_wellres_1[0] * alpha_wellres_1 + k / niter_wellres_1[0] * (1 + alpha_wellres_1)) for p, k in zip(nprocs, niter_wellres_1)]

    niter_wellres_2 = [9, 11, 16, 29]
    alpha_wellres_2 = 1.0 / 2.0
    speedup_wellres_2 = [p / (p / niter_wellres_2[0] * alpha_wellres_2 + k / niter_wellres_2[0] * (1 + alpha_wellres_2)) for p, k in zip(nprocs, niter_wellres_2)]

    plt_helper.setup_mpl()
    plt_helper.newfig(textwidth=238.96, scale=1.0)
    plt_helper.plt.semilogx(nprocs, speedup_wellres_1, color='r', marker='d', markersize=6, label=r'$Nx_\mathcal{F}=32, \alpha=\frac{1}{4}$')
    plt_helper.plt.semilogx(nprocs, speedup_wellres_2, color='b', marker='s', markersize=6, label=r'$Nx_\mathcal{F}=32, \alpha=\frac{1}{2}$')
    beautify_plot(nprocs, 'fool_speedup_wellres_iter')


if __name__ == '__main__':

    plot_data()