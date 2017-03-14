import matplotlib as mpl
mpl.use('pgf')
import matplotlib.pyplot as plt

import numpy as np


def figsize(textwidth, scale):
    fig_width_pt = textwidth                            # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27                         # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale    # width in inches
    fig_height = fig_width * golden_mean                # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


def setup_mpl():
    # Set up plotting parameters
    pgf_with_latex = {  # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
        "text.usetex": True,  # use LaTeX to write all text
        "font.family": "serif",
        "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": 8,  # LaTeX default is 10pt font.
        "axes.linewidth": 0.5,
        "font.size": 8,
        "legend.fontsize": 6,  # Make the legend/label fonts a little smaller
        "legend.numpoints": 1,
        "xtick.labelsize": 6,
        "xtick.major.width": 0.5,  # major tick width in points
        "xtick.minor.width": 0.25,
        "ytick.labelsize": 6,
        "ytick.major.width": 0.5,  # major tick width in points
        "ytick.minor.width": 0.25,
        "lines.markersize": 4,
        "lines.markeredgewidth": 0.5,
        "grid.linewidth": 0.5,
        "grid.linestyle": '-',
        "grid.alpha": 0.25,
        "figure.subplot.hspace": 0.0,
        "savefig.pad_inches": 0.01,
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts becasue your computer can handle it :)
            r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
        ]
    }
    mpl.rcParams.update(pgf_with_latex)


def newfig(textwidth, scale):
    plt.clf()
    fig = plt.figure(figsize=figsize(textwidth, scale))
    ax = fig.add_subplot()
    return fig, ax


def savefig(filename, save_pdf=True, save_pgf=True, save_png=True):
    if save_pgf:
        plt.savefig('{}.pgf'.format(filename), rasterized=True, bbox_inches='tight')
    if save_pdf:
        plt.savefig('{}.pdf'.format(filename), rasterized=True, bbox_inches='tight')
    if save_png:
        plt.savefig('{}.png'.format(filename), rasterized=True, bbox_inches='tight')
