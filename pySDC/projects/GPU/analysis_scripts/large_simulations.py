import numpy as np
from pySDC.projects.GPU.etc.generate_jobscript import write_jobscript

class LargeSim:
    config = None
    params = {}

    def setup_CPU_params(self):
        raise NotImplementedError()

    def setup_GPU_params(self):
        raise NotImplementedError()

    @property
    def plotting_params(self):
        return {
                'tasks_per_node': 20,
                'partition': 'develgpus',
                'cluster': 'jusuf',
                'time': '0:20:00',
        }


    def write_jobscript_for_run(self, submit=True):
        procs = self.params['procs']
        useGPU = self.params['useGPU']
        partition = self.params['partition']
        tasks_per_node = self.params['tasks_per_node']
        cluster = self.params['cluster']
        res = self.params['res']
        time = self.params['time']

        sbatch_options = [
            f'-n {np.prod(procs)}',
            f'-p {partition}',
            f'--tasks-per-node={tasks_per_node}',
            f'--time={time}',
        ]

        srun_options = [f'--tasks-per-node={tasks_per_node}']

        if useGPU:
            srun_options += ['--cpus-per-task=4', '--gpus-per-task=1']
            sbatch_options += ['--cpus-per-task=4', '--gpus-per-task=1']

        procs = (''.join(f'{me}/' for me in procs))[:-1]
        command = f'run_experiment.py --mode=run --res={res} --config={self.config} --procs={procs}'

        if useGPU:
            command += ' --useGPU=True'

        write_jobscript(sbatch_options, srun_options, command, cluster, submit=submit)

    def write_jobscript_for_plotting(self, num_procs=20, submit=True):
        procs_sim = self.params['procs']
        useGPU = self.params['useGPU']
        res = self.params['res']
        procs = num_procs
        partition = self.plotting_params['partition']
        tasks_per_node = self.plotting_params['tasks_per_node']
        cluster = self.plotting_params['cluster']

        sbatch_options = [
            f'-n {np.prod(procs)}',
            f'-p {partition}',
            f'--tasks-per-node={tasks_per_node}',
            f'--time={self.plotting_params["time"]}',
        ]

        srun_options = [f'--tasks-per-node={tasks_per_node}']

        procs_sim = (''.join(f'{me}/' for me in procs_sim))[:-1]
        command = f'run_experiment.py --mode=plot --res={res} --config={self.config} --procs={procs_sim}'

        if useGPU:
            command += ' --useGPU=True'

        write_jobscript(sbatch_options, srun_options, command, cluster, submit=submit)

    def write_jobscript_for_video(self, num_procs=20, submit=True):
        procs_sim = self.params['procs']
        useGPU = self.params['useGPU']
        res = self.params['res']
        procs = num_procs
        partition = self.plotting_params['partition']
        tasks_per_node = self.plotting_params['tasks_per_node']
        cluster = self.plotting_params['cluster']

        sbatch_options = [
            f'-n {np.prod(procs)}',
            f'-p {partition}',
            f'--tasks-per-node={tasks_per_node}',
            f'--time={self.plotting_params["time"]}',
        ]

        srun_options = [f'--tasks-per-node={tasks_per_node}']

        procs_sim = (''.join(f'{me}/' for me in procs_sim))[:-1]
        command = f'run_experiment.py --mode=video --res={res} --config={self.config} --procs={procs_sim}'

        if useGPU:
            command += ' --useGPU=True'

        write_jobscript(sbatch_options, srun_options, command, cluster, submit=submit)


class GSLarge(LargeSim):
    config = 'GS_large'

    def setup_CPU_params(self):
        """
        Test params with a small run.
        """
        self.params = {
                'procs': [1, 4, 4],
                'useGPU': False,
                'tasks_per_node': 16,
                'partition': 'develgpus',
                'cluster': 'jusuf',
                'res': 512,
                'time': '0:30:00',
        }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, default='GS')
    parser.add_argument('--XPU', type=str, choices=['CPU', 'GPU'], default='CPU')
    parser.add_argument('--mode', type=str, choices=['run', 'plot', 'video'], default='plot')
    parser.add_argument('--submit', type=str, choices=['yes', 'no'], default='yes')
    parser.add_argument('--num_procs', type=int, default=20)
    args = parser.parse_args()

    if args.problem == 'GS':
        cls = GSLarge
    else:
        raise NotImplementedError

    sim = cls()

    if args.XPU == 'CPU':
        sim.setup_CPU_params()
    elif args.XPU == 'GPU':
        sim.setup_GPU_params()
    else:
        raise NotImplementedError()

    if args.mode == 'run':
        sim.write_jobscript_for_run(submit=args.submit)
    elif args.mode == 'plot':
        sim.write_jobscript_for_plotting(num_procs=args.num_procs, submit=args.submit)
    elif args.mode == 'video':
        sim.write_jobscript_for_video(num_procs=args.num_procs, submit=args.submit)
    else:
        raise NotImplementedError()
