# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np


def matrix_plot(matrix,ax_x=False,ax_y=False):
    fig,ax = plt.subplots()
    plt_mat = ax.imshow(matrix, cmap=plt.cm.jet, interpolation='nearest')
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if ax_x is not False:
        plt.xticks([0,matrix.shape[0]/2, matrix.shape[0]],np.linspace(ax_x[0],ax_x[-1],3))
    if ax_y is not False:
        plt.yticks([0,matrix.shape[1]/2, matrix.shape[1]],np.linspace(ax_y[0],ax_y[-1],3))
    # colorbar
    plt.colorbar(plt_mat)
    plt.show()
