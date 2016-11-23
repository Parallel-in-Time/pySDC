import numpy as np
from matplotlib import pyplot as plt
from pylab import rcParams


def plot_buoyancy(cwd=''):
    """
    Plotting routine for the cross section of the buoyancy

    Args:
        cwd (string): current working directory
    """

    xx = np.load(cwd + 'xaxis.npy')
    uend = np.load(cwd + 'sdc.npy')
    udirk = np.load(cwd + 'dirk.npy')
    uimex = np.load(cwd + 'rkimex.npy')
    uref = np.load(cwd + 'uref.npy')
    usplit = np.load(cwd + 'split.npy')

    print("Estimated discretisation error split explicit:  %5.3e" %
          (np.linalg.norm(usplit.flatten() - uref.flatten(), np.inf) / np.linalg.norm(uref.flatten(), np.inf)))
    print("Estimated discretisation error of DIRK: %5.3e" %
          (np.linalg.norm(udirk.flatten() - uref.flatten(), np.inf) / np.linalg.norm(uref.flatten(), np.inf)))
    print("Estimated discretisation error of RK-IMEX:  %5.3e" %
          (np.linalg.norm(uimex.flatten() - uref.flatten(), np.inf) / np.linalg.norm(uref.flatten(), np.inf)))
    print("Estimated discretisation error of SDC:  %5.3e" %
          (np.linalg.norm(uend.flatten() - uref.flatten(), np.inf) / np.linalg.norm(uref.flatten(), np.inf)))

    fs = 8
    rcParams['figure.figsize'] = 5.0, 2.5
    plt.figure()
    plt.plot(xx[:, 5], udirk[2, :, 5], '--', color='g', markersize=fs - 2, label='DIRK(4)', dashes=(3, 3))
    plt.plot(xx[:, 5], uend[2, :, 5], '-', color='b', label='SDC(4)')
    plt.plot(xx[:, 5], uimex[2, :, 5], '--', color='r', markersize=fs - 2, label='IMEX(4)', dashes=(3, 3))
    plt.legend(loc='lower left', fontsize=fs, prop={'size': fs})
    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.xlabel('x [km]', fontsize=fs, labelpad=0)
    plt.ylabel('Bouyancy', fontsize=fs, labelpad=1)
    filename = 'boussinesq.png'
    plt.savefig(filename, bbox_inches='tight')


if __name__ == "__main__":
    plot_buoyancy()
