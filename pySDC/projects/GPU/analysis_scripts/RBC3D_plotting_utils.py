from pySDC.helpers.plot_helper import figsize_by_journal, setup_mpl
import warnings

setup_mpl()


def get_plotting_style(config):  # pragma: no cover

    args = {'color': None, 'ls': None, 'marker': None, 'markersize': 6, 'label': None}

    if config == 'RBC3DG4R4SDC23Ra1e5':
        args['color'] = 'tab:blue'
        args['ls'] = '-'
        args['marker'] = 'o'
        args['label'] = 'SDC23'
    elif config == 'RBC3DG4R4SDC34Ra1e5':
        args['color'] = 'tab:orange'
        args['ls'] = '-'
        args['marker'] = '<'
        args['label'] = 'SDC34'
    elif config in ['RBC3DG4R4SDC44Ra1e5', 'RBC3DG4R4Ra1e5']:
        args['color'] = 'tab:green'
        args['ls'] = '-'
        args['marker'] = 'x'
        args['label'] = 'SDC44'
    elif config == 'RBC3DG4R4EulerRa1e5':
        args['color'] = 'tab:purple'
        args['ls'] = '--'
        args['marker'] = '.'
        args['label'] = 'Euler'
    elif config == 'RBC3DG4R4RKRa1e5':
        args['color'] = 'tab:red'
        args['ls'] = '--'
        args['marker'] = '>'
        args['label'] = 'RK444'
    else:
        warnings.warn(f'No plotting style for {config=!r}')

    return args


def savefig(fig, name, format='pdf', base_path='./plots', **kwargs):  # pragma: no cover
    path = f'{base_path}/{name}.{format}'
    fig.savefig(path, bbox_inches='tight', **kwargs)
    print(f'Saved figure {path!r}')
