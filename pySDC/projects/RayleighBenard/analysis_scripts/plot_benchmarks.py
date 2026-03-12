import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pySDC.projects.RayleighBenard.analysis_scripts.plotting_utils import figsize, savefig, get_plotting_style

COLORS = {'JUSUF': 'tab:blue', 'BOOSTER': 'tab:orange', 'JUPITER': 'tab:green'}
RA_TO_RES = {
    '1e5': r'$N=128\times 128\times 32$',
    '1e6': r'$N=256\times 256\times 64$',
    '1e7': r'$N=512\times 512\times 128$',
    '1e8': r'$N=1024\times 1024\times 256$',
}


def get_path(path):
    from pathlib import Path

    return f'{Path(__file__).parent.parent}/{path}'


def plot_binding():  # pragma: no cover
    fig, axs = plt.subplots(1, 2, figsize=figsize(scale=1, ratio=0.4))
    all_data = [
        pd.read_csv(get_path(f'benchmarks/results/{machine}_RBC3DG4R4SDC44Ra{Ra}.txt'))
        for machine, Ra in zip(['JUSUF', 'JUPITER'], ['1e5', '1e7'], strict=True)
    ]

    for ax, data in zip(axs.flatten(), all_data, strict=True):
        binds = ['block:cyclic:cyclic', 'cyclic:cyclic:cyclic']
        dists = ['space_first', 'time_first']
        for dist in dists:
            dist_label = 'space-major' if dist[0] == 's' else 'time-major'
            ls = '-' if dist[0] == 's' else '--'
            for bind in binds:
                bind_label = 'block' if bind[0] == 'b' else 'cyclic'
                ms = '.' if bind[0] == 'b' else 'x'
                color = 'tab:blue' if bind[0] == 'b' else 'tab:orange'
                mask = np.logical_and(data.distribution == bind, data.binding == dist)
                mask = np.logical_and(mask, data.ntasks_time > 1)
                ax.loglog(
                    data.procs[mask],
                    data.time[mask],
                    label=f'{dist_label}, {bind_label}',
                    ls=ls,
                    marker=ms,
                    color=color,
                )
        ax.legend(frameon=False)
        XPU = 'GPU' if any(np.isfinite(data.time_GPU)) else 'CPU'
        _res = data.res[0]
        ax.set_title(fr'$N={{{_res*4}}}\times{{{_res*4}}} \times {{{_res}}}$ {XPU}')
        ax.set_xlabel(r'$N_\mathrm{tasks}$')
        ax.set_ylabel('time / s')
    fig.tight_layout()
    savefig(fig, 'pySDC_binding')


def compare_methods_single_config(
    ax, machine, Ra='1e5', normalize=False, scale_by_stability_limit=False
):  # pragma: no cover
    methods = ['SDC44', 'SDC23', 'RK', 'Euler']

    assert not (normalize and scale_by_stability_limit)

    stable_step_sizes = {}
    stable_step_sizes['1e5'] = {'SDC44': 0.06, 'SDC23': 0.06, 'RK': 0.05, 'Euler': 0.02}
    stable_step_sizes['1e6'] = {'SDC44': 0.01, 'SDC23': 0.01, 'RK': 0.01, 'Euler': 0.005}
    stable_step_sizes['1e7'] = {'SDC44': 0.005, 'SDC23': 0.005, 'RK': 0.004, 'Euler': 0.001}

    if normalize:
        norm_data = pd.read_csv(get_path(f'benchmarks/results/{machine}_RBC3DG4R4EulerRa{Ra}.txt'))
        for cost in [3, 4, 5, 13]:
            ax.axhline(cost, ls=':', color='black')
    else:
        norm_data = None

    for method in methods:
        config = f"RBC3DG4R4{method}Ra{Ra}"
        data = pd.read_csv(get_path(f'benchmarks/results/{machine}_{config}.txt'))
        bind_mask = data.distribution == 'block:cyclic:cyclic'
        dist_mask = data.binding == 'time_first'
        base_mask = np.logical_and(bind_mask, dist_mask)

        mask = base_mask

        for _tasks_time in np.unique(data.ntasks_time):
            mask = np.logical_and(data.ntasks_time == _tasks_time, base_mask)
            plotting_style = get_plotting_style(config)
            plotting_style['ls'] = '-' if _tasks_time == 1 else '--'
            plotting_style['label'] += ' PinT' if _tasks_time > 1 else ''

            timings = np.array(data.time[mask])
            procs = np.array(data.procs[mask])
            space_procs = np.array(data.ntasks_space[mask])

            if scale_by_stability_limit:
                timings *= 1 / stable_step_sizes[Ra][method]

            if norm_data is not None:
                for i in range(len(timings)):
                    ref_time = np.array(norm_data.time[norm_data.ntasks_space == space_procs[i]])[0]
                    timings[i] /= ref_time
                # print(f'{machine} {config} {_tasks_time} {np.min(data.time[mask]) / np.min(norm_data.time):.2f}')
            ax.loglog(procs, timings, **plotting_style)

    if normalize:
        ax.set_ylabel(r'time / ($t_\mathrm{E}+t_\mathrm{S}$)')
    elif scale_by_stability_limit:
        ax.set_ylabel(r'free fall time / s')
    else:
        ax.set_ylabel(r'time per step / s')

    ax.set_xlabel(r'$N_\mathrm{tasks}$')
    ax.set_title(RA_TO_RES[Ra])
    # ax.legend(frameon=False)


def compare_methods(machine, normalize=False, scale_by_stability_limit=False):  # pragma: no cover
    fig, axs = plt.subplots(1, 2, figsize=figsize(scale=1, ratio=0.4))

    Ras = {'JUSUF': ['1e5', '1e6'], 'BOOSTER': ['1e6', '1e7'], 'JUPITER': ['1e6', '1e7']}
    for Ra, ax in zip(Ras[machine], axs.flatten(), strict=True):
        compare_methods_single_config(
            ax, machine, Ra, normalize=normalize, scale_by_stability_limit=scale_by_stability_limit
        )

    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=True))  # removes duplicates

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),  # centered below figure
        ncol=4,
        frameon=False,
    )
    fig.tight_layout()
    if normalize:
        savefig(fig, f"compare_methods_{machine}_normalized")
    elif scale_by_stability_limit:
        savefig(fig, f"compare_methods_{machine}_scaled_by_stability")
    else:
        savefig(fig, f"compare_methods_{machine}")


def plot_space_scaling(method='Euler'):  # pragma: no cover
    fig, axs = plt.subplots(1, 3, figsize=figsize(scale=1, ratio=0.4), sharex=True, sharey=True)

    for machine in ['JUSUF', 'BOOSTER', 'JUPITER']:
        for Ra, ax in zip(['1e5', '1e6', '1e7'], axs.flatten(), strict=True):
            if machine == 'JUPITER' and Ra == '1e5':
                continue
            data = pd.read_csv(get_path(f'benchmarks/results/{machine}_RBC3DG4R4{method}Ra{Ra}.txt'))

            mask = np.isfinite(data.time)
            time = np.array(data.time[mask])
            procs = np.array(data.procs[mask])

            time_max = time[0]
            time_min = np.min(time)
            procs_max = procs[0]
            procs_min = procs_max * time_max / time_min
            ax.loglog([procs_max, procs_min], [time_max, time_min], color='black', ls=':', label='ideal')

            ax.loglog(procs, time, color=COLORS[machine], label=f'{machine}')
            ax.set_title(RA_TO_RES[Ra])

    for ax in axs.flatten():
        ax.set_xlabel(r'$N_\mathrm{tasks}$')
        ax.set_box_aspect(1)
    axs[0].set_ylabel(r'time / s')

    handles, labels = axs[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=True))  # removes duplicates

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),  # centered below figure
        ncol=4,
        frameon=False,
    )
    fig.tight_layout()
    savefig(fig, f'space_scaling_{method}')


def plot_space_time_scaling(method='SDC44'):  # pragma: no cover
    fig, axs = plt.subplots(1, 3, figsize=figsize(scale=1, ratio=0.4), sharex=True, sharey=True)
    PinT_efficiency_fig, PinT_efficiency_ax = plt.subplots(figsize=figsize(scale=1, ratio=0.4))

    for machine in ['JUSUF', 'BOOSTER', 'JUPITER']:
        for Ra, ax in zip(['1e5', '1e6', '1e7'], axs.flatten(), strict=True):
            data = pd.read_csv(get_path(f'benchmarks/results/{machine}_RBC3DG4R4{method}Ra{Ra}.txt'))

            base_mask = np.logical_and(data.distribution == 'block:cyclic:cyclic', data.binding == 'time_first')
            base_mask = np.logical_and(base_mask, np.isfinite(data.time))

            for ntasks_time in np.unique(data.ntasks_time):
                mask = np.logical_and(base_mask, data.ntasks_time == ntasks_time)
                time = np.array(data.time[mask])
                procs = np.array(data.procs[mask])

                plotting_style = {}
                plotting_style['ls'] = '-' if ntasks_time == 1 else '--'
                plotting_style['label'] = machine + (' PinT' if ntasks_time > 1 else '')
                plotting_style['color'] = COLORS[machine]

                time_max = time[0]
                time_min = np.min(time)
                procs_max = procs[0]
                procs_min = procs_max * time_max / time_min
                ax.loglog([procs_max, procs_min], [time_max, time_min], color='black', ls=':', label='ideal')

                ax.loglog(procs, time, **plotting_style)
                ax.set_title(RA_TO_RES[Ra])

                if ntasks_time > 1:  # plot PinT efficiency
                    ntasks_space = procs // ntasks_time
                    time_s = np.array(
                        [
                            np.nanmin(
                                np.array(
                                    data.time[np.logical_and(data.ntasks_time == 1, data.ntasks_space == _ntasks_space)]
                                )
                            )
                            for _ntasks_space in ntasks_space
                        ]
                    )
                    plotting_style = {}
                    plotting_style['color'] = COLORS[machine]
                    plotting_style['marker'] = {'1e5': '.', '1e6': 'x', '1e7': 'o'}[Ra]
                    plotting_style['ls'] = {'JUSUF': '-', 'BOOSTER': '-.', 'JUPITER': '--'}[machine]
                    plotting_style['label'] = f'{RA_TO_RES[Ra]} {machine}'
                    PinT_efficiency_ax.plot(ntasks_space, time_s / time / ntasks_time, **plotting_style)
                    PinT_efficiency_ax.set_xscale('log')

    for ax in axs.flatten():
        ax.set_xlabel(r'$N_\mathrm{tasks}$')
        ax.set_box_aspect(1)
    axs[0].set_ylabel(r'time / s')

    handles, labels = axs[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=True))  # removes duplicates

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),  # centered below figure
        ncol=4,
        frameon=False,
    )
    fig.tight_layout()
    savefig(fig, f'space_time_scaling_{method}')

    handles, labels = PinT_efficiency_ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=True))  # removes duplicates

    PinT_efficiency_ax.set_ylabel('PinT Efficiency')
    PinT_efficiency_ax.set_xlabel(r'$N_\mathrm{tasks,\ space}$')
    PinT_efficiency_fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.4),  # centered below figure
        ncol=2,
        frameon=False,
    )
    PinT_efficiency_fig.tight_layout()
    savefig(PinT_efficiency_fig, f'PinT_efficiency_{method}')


def make_plots_for_SIAMPP26():  # pragma: no cover
    fig, axs = plt.subplots(1, 2, figsize=figsize(scale=0.9, ratio=0.4), sharex=True, sharey=True)

    for machine in ['JUSUF', 'JUPITER']:
        for Ra, ax in zip(['1e6', '1e7'], axs.flatten(), strict=True):
            data = pd.read_csv(get_path(f'benchmarks/results/{machine}_RBC3DG4R4SDC44Ra{Ra}.txt'))

            base_mask = np.logical_and(data.distribution == 'block:cyclic:cyclic', data.binding == 'time_first')
            base_mask = np.logical_and(base_mask, np.isfinite(data.time))

            for ntasks_time in np.unique(data.ntasks_time):
                mask = np.logical_and(base_mask, data.ntasks_time == ntasks_time)
                time = np.array(data.time[mask])
                procs = np.array(data.procs[mask])

                plotting_style = {}
                plotting_style['ls'] = '-' if ntasks_time == 1 else '--'
                plotting_style['label'] = machine + (' PinT' if ntasks_time > 1 else '')
                plotting_style['color'] = COLORS[machine]

                time_max = time[0]
                time_min = np.min(time)
                procs_max = procs[0]
                procs_min = procs_max * time_max / time_min
                ax.loglog([procs_max, procs_min], [time_max, time_min], color='black', ls=':', label='ideal')

                ax.loglog(procs, time, **plotting_style)
                ax.set_title(f'Ra={Ra}\n{RA_TO_RES[Ra]}')

    for ax in axs.flatten():
        ax.set_xlabel(r'$N_\mathrm{tasks}$')
    axs[0].set_ylabel(r'time / s')

    handles, labels = axs[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=True))  # removes duplicates

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),  # centered below figure
        ncol=3,
        frameon=False,
    )
    fig.tight_layout()
    savefig(fig, 'space_time_scaling_SIAMPP')

    # comparing methods
    fig, axs = plt.subplots(1, 2, figsize=figsize(scale=0.9, ratio=0.4))

    for scale_by_stability_limit, ax in zip([False, True], axs.flatten(), strict=True):
        compare_methods_single_config(
            ax, 'JUPITER', '1e7', normalize=False, scale_by_stability_limit=scale_by_stability_limit
        )
        ax.set_title(f'Ra={Ra}\n{RA_TO_RES[Ra]}')

    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=True))  # removes duplicates

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),  # centered below figure
        ncol=3,
        frameon=False,
    )
    fig.tight_layout()
    savefig(fig, "compare_methods_SIAMPP")


def make_plots_for_paper():  # pragma: no cover
    plot_space_scaling()
    plot_binding()
    plot_space_time_scaling('SDC44')
    compare_methods('JUPITER', scale_by_stability_limit=False)
    compare_methods('JUPITER', scale_by_stability_limit=True)


if __name__ == '__main__':
    make_plots_for_paper()
    plt.show()
