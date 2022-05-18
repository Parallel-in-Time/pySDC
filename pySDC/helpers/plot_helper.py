import matplotlib as mpl
import matplotlib.pyplot as plt
from distutils.spawn import find_executable

default_mpl_params = mpl.rcParams.copy()


def figsize(textwidth, scale, ratio):
    fig_width_pt = textwidth                            # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27                         # Convert pt to inch
    fig_width = fig_width_pt * inches_per_pt * scale    # width in inches
    fig_height = fig_width * ratio                # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


def setup_mpl(font_size=8, reset=False):
    if reset:
        mpl.rcParams.update(default_mpl_params)

    # Set up plotting parameters
    style_options = {  # setup matplotlib to use latex for output
        "font.family": "serif",
        "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
        "font.sans-serif": [],
        "font.monospace": [],
        # "axes.labelsize": 8,  # LaTeX default is 10pt font.
        "axes.linewidth": 0.5,
        "font.size": font_size,
        # "legend.fontsize": 6,  # Make the legend/label fonts a little smaller
        "legend.numpoints": 1,
        # "xtick.labelsize": 6,
        "xtick.major.width": 0.5,  # major tick width in points
        "xtick.minor.width": 0.25,
        # "ytick.labelsize": 6,
        "ytick.major.width": 0.5,  # major tick width in points
        "ytick.minor.width": 0.25,
        "lines.markersize": 4,
        "lines.markeredgewidth": 0.5,
        "grid.linewidth": 0.5,
        "grid.linestyle": '-',
        "grid.alpha": 0.25,
        "figure.subplot.hspace": 0.0,
        "savefig.pad_inches": 0.01,
    }

    mpl.rcParams.update(style_options)

    if find_executable('latex'):

        latex_support = {
            "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
            "text.usetex": True,  # use LaTeX to write all text
            "pgf.preamble": r"\usepackage[utf8x]{inputenc}"
                            r"\usepackage[T1]{fontenc}"
                            r"\usepackage{underscore}"
                            r"\usepackage{amsmath,amssymb,marvosym}"
        }
    else:
        latex_support = {
            "text.usetex": False,  # use LaTeX to write all text
        }

    mpl.rcParams.update(latex_support)


def newfig(textwidth, scale, ratio=0.6180339887):
    plt.clf()
    fig, ax = plt.subplots(figsize=figsize(textwidth, scale, ratio))
    return fig, ax


def savefig(filename, save_pdf=True, save_pgf=True, save_png=True):
    if save_pgf and find_executable('latex'):
        plt.savefig('{}.pgf'.format(filename), bbox_inches='tight')
    if save_pdf:
        plt.savefig('{}.pdf'.format(filename), bbox_inches='tight')
    if save_png:
        plt.savefig('{}.png'.format(filename), bbox_inches='tight')
    plt.close()
