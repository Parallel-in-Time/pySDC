import matplotlib.pyplot as plt
import numpy as np
import pickle
from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.GPU.configs.base_config import get_config
from pySDC.projects.GPU.etc.generate_jobscript import write_jobscript, PROJECT_PATH


class ScalingConfig(object):
    cluster = None
    config = ''
    base_resolution = -1
    base_resolution_weak = -1
    useGPU = False
    partition = None
    tasks_per_node = None
    ndim = 2
    tasks_time = 1
    max_steps_space = None
    max_steps_space_weak = None
    sbatch_options = []

    def __init__(self, space_time_parallel):
        if space_time_parallel in ['False', False]:
            self._tasks_time = 1
        else:
            self._tasks_time = self.tasks_time

    def get_resolution_and_tasks(self, strong, i):
        if strong:
            return self.base_resolution, [1, self._tasks_time, 2**i]
        else:
            return self.base_resolution_weak * (2**i), [1, self._tasks_time, (2 * self.ndim) ** i]

    def run_scaling_test(self, strong=True):
        max_steps = self.max_steps_space if strong else self.max_steps_space_weak
        for i in range(max_steps):
            res, procs = self.get_resolution_and_tasks(strong, i)

            sbatch_options = [f'-n {np.prod(procs)}', f'-p {self.partition}'] + self.sbatch_options
            if self.useGPU:
                srun_options = ['--cpus-per-task=4', '--gpus-per-task=1'] + self.sbatch_options
                sbatch_options += ['--cpus-per-task=4', '--gpus-per-task=1']
            else:
                srun_options = []

            procs = (''.join(f'{me}/' for me in procs))[:-1]
            command = f'run_experiment.py --mode=run --res={res} --config={self.config} --procs={procs}'

            if self.useGPU:
                command += ' --useGPU=True'

            write_jobscript(sbatch_options, srun_options, command, self.cluster)

    def plot_scaling_test(self, strong, ax, plot_ideal=False, **plotting_params):  # pragma: no cover
        timings = {}

        max_steps = self.max_steps_space if strong else self.max_steps_space_weak
        for i in range(max_steps):
            res, procs = self.get_resolution_and_tasks(strong, i)

            args = {'useGPU': self.useGPU, 'config': self.config, 'res': res, 'procs': procs, 'mode': None}

            config = get_config(args)

            path = f'data/{config.get_path(ranks=[me -1 for me in procs])}-stats-whole-run.pickle'
            try:
                with open(path, 'rb') as file:
                    stats = pickle.load(file)

                timing_step = get_sorted(stats, type='timing_step')

                timings[np.prod(procs) / self.tasks_per_node] = np.mean([me[1] for me in timing_step])
            except FileNotFoundError:
                pass

        ax.loglog(timings.keys(), timings.values(), **plotting_params)
        if plot_ideal:
            ax.loglog(
                timings.keys(),
                list(timings.values())[0] * list(timings.keys())[0] / np.array(list(timings.keys())),
                ls='--',
                color='grey',
                label='ideal',
            )
        ax.set_xlabel(r'$N_\mathrm{nodes}$')
        ax.set_ylabel(r'$t_\mathrm{step}$')


class CPUConfig(ScalingConfig):
    cluster = 'jusuf'
    partition = 'batch'
    tasks_per_node = 16
    sbatch_options = ['--tasks-per-node=16']


class GPUConfig(ScalingConfig):
    cluster = 'booster'
    partition = 'booster'
    tasks_per_node = 4
    useGPU = True


class GrayScottSpaceScalingCPU(CPUConfig, ScalingConfig):
    base_resolution = 4096
    base_resolution_weak = 256
    config = 'GS_scaling'
    max_steps_space = 10
    max_steps_space_weak = 6
    tasks_time = 3


class GrayScottSpaceScalingGPU(GPUConfig, ScalingConfig):
    base_resolution_weak = 256 * 2
    base_resolution = 4096
    config = 'GS_scaling'
    max_steps_space = 6
    max_steps_space_weak = 4
    tasks_time = 3


def plot_scalings(strong, problem, kwargs):  # pragma: no cover
    if problem == 'GS':
        fig, ax = plt.subplots()

        plottings_params = [
            {'plot_ideal': strong, 'marker': 'x', 'label': 'CPU'},
            {'marker': '>', 'label': 'CPU space time parallel'},
            {'marker': '^', 'label': 'GPU'},
        ]
        configs = [
            GrayScottSpaceScalingCPU(space_time_parallel=False),
            GrayScottSpaceScalingCPU(space_time_parallel=True),
            GrayScottSpaceScalingGPU(space_time_parallel=False),
        ]

        for config, params in zip(configs, plottings_params):
            config.plot_scaling_test(strong=strong, ax=ax, **params)
        ax.legend(frameon=False)
        fig.savefig(f'{PROJECT_PATH}/plots/{"strong" if strong else "weak"}_scaling_{problem}.pdf', bbox_inches='tight')
    else:
        raise NotImplementedError


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--scaling', type=str, choices=['strong', 'weak'], default='strong')
    parser.add_argument('--mode', type=str, choices=['run', 'plot'], default='run')
    parser.add_argument('--problem', type=str, default='GS')
    parser.add_argument('--XPU', type=str, choices=['CPU', 'GPU'], default='CPU')
    parser.add_argument('--space_time', type=str, choices=['True', 'False'], default='False')

    args = parser.parse_args()

    strong = args.scaling == 'strong'

    if args.problem == 'GS':
        if args.XPU == 'CPU':
            configClass = GrayScottSpaceScalingCPU
        else:
            configClass = GrayScottSpaceScalingGPU
    else:
        raise NotImplementedError(f'Don\'t know problem {args.problem!r}')

    kwargs = {'space_time_parallel': args.space_time}
    config = configClass(**kwargs)

    if args.mode == 'run':
        config.run_scaling_test(strong=strong)
    elif args.mode == 'plot':
        plot_scalings(strong=strong, problem=args.problem, kwargs=kwargs)
    else:
        raise NotImplementedError(f'Don\'t know mode {args.mode!r}')
