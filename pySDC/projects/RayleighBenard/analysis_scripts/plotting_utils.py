from functools import partial
import warnings
import os
import matplotlib.pyplot as plt

from pySDC.helpers.plot_helper import figsize_by_journal, setup_mpl

setup_mpl()
plt.rcParams['markers.fillstyle'] = 'none'

figsize = partial(figsize_by_journal, journal='Nature_CS')


def get_plotting_style(config):  # pragma: no cover

    args = {'color': None, 'ls': None, 'marker': None, 'markersize': 6, 'label': None}

    config_no_Ra = config[: config.index('Ra')]

    if config_no_Ra == 'RBC3DG4R4SDC22':
        args['color'] = 'tab:brown'
        args['ls'] = '-'
        args['marker'] = '3'
        args['label'] = 'SDC22'
    elif config_no_Ra == 'RBC3DG4R4SDC23':
        args['color'] = 'tab:blue'
        args['ls'] = '-'
        args['marker'] = 'o'
        args['label'] = 'SDC23'
    elif config_no_Ra == 'RBC3DG4R4SDC34':
        args['color'] = 'tab:orange'
        args['ls'] = '-'
        args['marker'] = '<'
        args['label'] = 'SDC34'
    elif config_no_Ra == 'RBC3DG4R4SDC44':
        args['color'] = 'tab:green'
        args['ls'] = '-'
        args['marker'] = 'x'
        args['label'] = 'SDC44'
    elif config_no_Ra == 'RBC3DG4R4Euler':
        args['color'] = 'tab:purple'
        args['ls'] = '--'
        args['marker'] = '.'
        args['label'] = 'RK111'
    elif config_no_Ra == 'RBC3DG4R4RK':
        args['color'] = 'tab:red'
        args['ls'] = '--'
        args['marker'] = '>'
        args['label'] = 'RK443'
    else:
        warnings.warn(f'No plotting style for {config=!r}')

    return args


def savefig(fig, name, format='pdf', base_path='./plots', **kwargs):  # pragma: no cover
    os.makedirs(base_path, exist_ok=True)

    path = f'{base_path}/{name}.{format}'
    fig.savefig(path, bbox_inches='tight', **kwargs)
    print(f'Saved figure {path!r}')
