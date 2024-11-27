import matplotlib.pyplot as plt
import numpy as np
import pickle
from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.GPU.configs.base_config import get_config
from pySDC.projects.GPU.etc.generate_jobscript import write_jobscript, PROJECT_PATH
from pySDC.helpers.plot_helper import setup_mpl, figsize_by_journal

setup_mpl()


class Experiment(object):
    def __init__(self, res, PinT=False, start=0, stop=-1, sequence=None, marker='x'):
        self.res = res
        self.PinT = PinT
        self.start = start
        self.stop = stop
        self._sequence = sequence
        self.marker = marker

    @property
    def sequence(self):
        if self._sequence is not None:
            return self._sequence
        else:
            sequence = []
            i = self.start
            while i <= self.stop:
                sequence += [i]
                i *= 2
            return sequence


class ScalingConfig(object):
    cluster = None
    config = ''
    useGPU = False
    partition = None
    tasks_per_node = None
    ndim = 2
    tasks_time = 1
    sbatch_options = []
    experiments = []

    def run_scaling_test(self, **kwargs):
        for experiment in self.experiments:
            res = experiment.res

            tasks_time = self.tasks_time if experiment.PinT else 1

            for i in experiment.sequence:
                procs = [1, tasks_time, int(np.ceil(i / tasks_time))]

                sbatch_options = [
                    f'-n {np.prod(procs)}',
                    f'-p {self.partition}',
                    f'--tasks-per-node={self.tasks_per_node}',
                ] + self.sbatch_options
                srun_options = [f'--tasks-per-node={self.tasks_per_node}']
                if self.useGPU:
                    srun_options += ['--cpus-per-task=4', '--gpus-per-task=1']
                    sbatch_options += ['--cpus-per-task=4', '--gpus-per-task=1']

                procs = (''.join(f'{me}/' for me in procs))[:-1]
                command = f'run_experiment.py --mode=run --res={res} --config={self.config} --procs={procs}'

                if self.useGPU:
                    command += ' --useGPU=True'

                write_jobscript(
                    sbatch_options, srun_options, command, self.cluster, name=f'{type(self).__name__}_{res}', **kwargs
                )

    def plot_scaling_test(self, ax, quantity='time', **plotting_params):  # pragma: no cover
        from matplotlib.colors import TABLEAU_COLORS

        cmap = TABLEAU_COLORS
        colors = list(cmap.values())

        for experiment in self.experiments:
            tasks_time = self.tasks_time if experiment.PinT else 1
            timings = {}

            plotting_params = {
                (False, False): {
                    'ls': '--',
                    'label': f'CPU ${{{experiment.res}}}^{{{self.ndim}}}$',
                    'color': colors[2],
                },
                (False, True): {
                    'ls': '-.',
                    'label': f'CPU PinT ${{{experiment.res}}}^{{{self.ndim}}}$',
                    'color': colors[3],
                },
                (True, False): {
                    'ls': '-',
                    'label': f'GPU ${{{experiment.res}}}^{{{self.ndim}}}$',
                    'color': colors[0],
                },
                (True, True): {
                    'ls': ':',
                    'label': f'GPU PinT ${{{experiment.res}}}^{{{self.ndim}}}$',
                    'color': colors[1],
                },
            }

            i = experiment.start
            res = experiment.res

            for i in experiment.sequence:
                procs = [1, tasks_time, int(np.ceil(i / tasks_time))]
                args = {'useGPU': self.useGPU, 'config': self.config, 'res': res, 'procs': procs, 'mode': None}

                config = get_config(args)

                path = f'data/{config.get_path(ranks=[me -1 for me in procs])}-stats-whole-run.pickle'
                try:
                    with open(path, 'rb') as file:
                        stats = pickle.load(file)

                    if args['useGPU']:
                        timing_step = get_sorted(stats, type='GPU_timing_step')
                    else:
                        timing_step = get_sorted(stats, type='timing_step')

                    t_mean = np.mean([me[1] for me in timing_step])

                    if quantity == 'throughput':
                        timings[np.prod(procs) / self.tasks_per_node] = experiment.res**self.ndim / t_mean
                    elif quantity == 'efficiency':
                        timings[np.prod(procs) / self.tasks_per_node] = (
                            experiment.res**self.ndim / t_mean / np.prod(procs)
                        )
                    elif quantity == 'time':
                        timings[np.prod(procs) / self.tasks_per_node] = t_mean
                    else:
                        raise NotImplementedError
                except (FileNotFoundError, ValueError):
                    pass

            ax.loglog(
                timings.keys(),
                timings.values(),
                **plotting_params[(self.useGPU, experiment.PinT)],
                marker=experiment.marker,
            )
        ax.set_xlabel(r'$N_\mathrm{nodes}$')
        labels = {
            'throughput': 'throughput / DoF/s',
            'time': r'$t_\mathrm{step}$ / s',
            'efficiency': 'efficiency / DoF/s/task',
        }
        ax.set_ylabel(labels[quantity])


class CPUConfig(ScalingConfig):
    # cluster = 'jusuf'
    cluster = 'juwels'
    partition = 'batch'
    tasks_per_node = 16


class GPUConfig(ScalingConfig):
    cluster = 'booster'
    partition = 'booster'
    tasks_per_node = 4
    useGPU = True


class GrayScottSpaceScalingCPU3D(CPUConfig, ScalingConfig):
    ndim = 3
    config = 'GS_scaling3D'
    tasks_time = 4
    sbatch_options = ['--time=1:30:00']
    experiments = [
        Experiment(res=512, PinT=False, start=1, stop=256, marker='>'),
        Experiment(res=512, PinT=True, start=1, stop=1024, marker='>'),
        Experiment(res=1024, PinT=False, start=256, stop=512, marker='x'),
        Experiment(res=1024, PinT=True, start=256, stop=2048, marker='x'),
        # {'res': 2304, 'PinT': True, 'start': 768, 'stop': 6144, 'marker': '.'},
        # {'res': 2304, 'PinT': False, 'start': 768, 'stop': 6144, 'marker': '.'},
    ]


class GrayScottSpaceScalingGPU3D(GPUConfig, ScalingConfig):
    ndim = 3
    config = 'GS_scaling3D'
    tasks_time = 4
    sbatch_options = ['--time=0:07:00']

    experiments = [
        Experiment(res=512, PinT=True, start=1, stop=512, marker='>'),
        Experiment(res=512, PinT=False, start=1, stop=256, marker='>'),
        Experiment(res=1024, PinT=True, start=16, stop=512, marker='x'),
        Experiment(res=1024, PinT=False, start=16, stop=1024, marker='x'),
        Experiment(res=2304, PinT=True, start=768, stop=1536, marker='.'),
        Experiment(res=2304, PinT=False, start=192, stop=768, marker='.'),
        # Experiment(res=4608, PinT=False, start=1536, stop=1536, marker='o'),
        Experiment(res=4480, PinT=True, sequence=[3584], marker='o'),
        Experiment(res=4480, PinT=False, sequence=[1494], marker='o'),
    ]


class RayleighBenardSpaceScalingCPU(CPUConfig, ScalingConfig):
    ndim = 2
    config = 'RBC_scaling'
    tasks_time = 4
    sbatch_options = ['--time=1:25:00']

    experiments = [
        Experiment(res=512, PinT=True, start=4, stop=512, marker='x'),
        Experiment(res=512, PinT=False, start=1, stop=128, marker='x'),
        Experiment(res=1024, PinT=True, start=4, stop=1024, marker='x'),
        Experiment(res=1024, PinT=False, start=1, stop=128, marker='x'),
        Experiment(res=2048, PinT=True, start=128, stop=1024, marker='.'),
        Experiment(res=2048, PinT=False, start=128, stop=256, marker='.'),
    ]


class RayleighBenardSpaceScalingGPU(GPUConfig, ScalingConfig):
    ndim = 3
    config = 'RBC_scaling'
    tasks_time = 4
    sbatch_options = ['--time=0:15:00']

    experiments = [
        Experiment(res=1024, PinT=True, start=4, stop=128, marker='x'),
        Experiment(res=1024, PinT=False, start=1, stop=32, marker='x'),
        Experiment(res=2048, PinT=True, start=16, stop=128, marker='.'),
        Experiment(res=2048, PinT=False, start=16, stop=128, marker='.'),
    ]


# class RayleighBenardSpaceScalingCPU(CPUConfig, ScalingConfig):
#     base_resolution = 1024
#     base_resolution_weak = 128
#     config = 'RBC_scaling'
#     max_steps_space = 13
#     max_steps_space_weak = 10
#     tasks_time = 4
#     max_nodes = 64
#     # sbatch_options = ['--time=3:30:00']
#     max_tasks = 256
#
#
# class RayleighBenardSpaceScalingGPU(GPUConfig, ScalingConfig):
#     base_resolution_weak = 256
#     base_resolution = 1024
#     config = 'RBC_scaling'
#     max_steps_space = 9
#     max_steps_space_weak = 9
#     tasks_time = 4
#     max_tasks = 256
#     sbatch_options = ['--time=0:15:00']
#     max_nodes = 64


class RayleighBenardDedalusComparison(CPUConfig, ScalingConfig):
    base_resolution = 256
    config = 'RBC_Tibo'
    max_steps_space = 6
    tasks_time = 4


class RayleighBenardDedalusComparisonGPU(GPUConfig, ScalingConfig):
    base_resolution_weak = 256
    base_resolution = 256
    config = 'RBC_Tibo'
    max_steps_space = 4
    max_steps_space_weak = 4
    tasks_time = 4


def plot_scalings(problem, **kwargs):  # pragma: no cover
    if problem == 'GS':
        configs = [
            GrayScottSpaceScalingCPU(),
            GrayScottSpaceScalingGPU(),
        ]
    elif problem == 'GS3D':
        configs = [
            GrayScottSpaceScalingCPU3D(),
            GrayScottSpaceScalingGPU3D(),
        ]
    elif problem == 'RBC':
        configs = [
            RayleighBenardSpaceScalingCPU(),
            RayleighBenardSpaceScalingGPU(),
        ]
    elif problem == 'RBC_dedalus':
        configs = [
            RayleighBenardDedalusComparison(),
            RayleighBenardDedalusComparisonGPU(),
        ]

    else:
        raise NotImplementedError

    ideal_lines = {
        ('GS3D', 'throughput'): {'x': [0.25, 400], 'y': [5e6, 8e9]},
        ('GS3D', 'time'): {'x': [0.25, 400], 'y': [80, 5e-2]},
    }

    fig, ax = plt.subplots(figsize=figsize_by_journal('TUHH_thesis', 1, 0.6))
    configs[1].plot_scaling_test(ax=ax, quantity='efficiency')
    ax.legend(frameon=False)
    path = f'{PROJECT_PATH}/plots/scaling_{problem}_efficiency.pdf'
    fig.savefig(path, bbox_inches='tight')
    print(f'Saved {path!r}', flush=True)

    for quantity in ['throughput', 'time']:
        fig, ax = plt.subplots(figsize=figsize_by_journal('TUHH_thesis', 1, 0.6))
        for config in configs:
            config.plot_scaling_test(ax=ax, quantity=quantity)
        if (problem, quantity) in ideal_lines.keys():
            ax.loglog(*ideal_lines[(problem, quantity)].values(), color='black', ls=':', label='ideal')
        ax.legend(frameon=False)
        path = f'{PROJECT_PATH}/plots/scaling_{problem}_{quantity}.pdf'
        fig.savefig(path, bbox_inches='tight')
        print(f'Saved {path!r}', flush=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['run', 'plot'], default='run')
    parser.add_argument('--problem', type=str, default='GS')
    parser.add_argument('--XPU', type=str, choices=['CPU', 'GPU'], default='CPU')
    parser.add_argument('--space_time', type=str, choices=['True', 'False'], default='False')
    parser.add_argument('--submit', type=str, choices=['True', 'False'], default='True')
    parser.add_argument('--nsys_profiling', type=str, choices=['True', 'False'], default='False')

    args = parser.parse_args()

    submit = args.submit == 'True'
    nsys_profiling = args.nsys_profiling == 'True'

    if args.problem == 'GS':
        if args.XPU == 'CPU':
            configClass = GrayScottSpaceScalingCPU
        else:
            configClass = GrayScottSpaceScalingGPU
    elif args.problem == 'GS3D':
        if args.XPU == 'CPU':
            configClass = GrayScottSpaceScalingCPU3D
        else:
            configClass = GrayScottSpaceScalingGPU3D
    elif args.problem == 'RBC':
        if args.XPU == 'CPU':
            configClass = RayleighBenardSpaceScalingCPU
        else:
            configClass = RayleighBenardSpaceScalingGPU
    elif args.problem == 'RBC_dedalus':
        if args.XPU == 'CPU':
            configClass = RayleighBenardDedalusComparison
        else:
            configClass = RayleighBenardDedalusComparisonGPU
    else:
        raise NotImplementedError(f'Don\'t know problem {args.problem!r}')

    config = configClass()

    if args.mode == 'run':
        config.run_scaling_test(submit=submit, nsys_profiling=nsys_profiling)
    elif args.mode == 'plot':
        plot_scalings(problem=args.problem)
    else:
        raise NotImplementedError(f'Don\'t know mode {args.mode!r}')
