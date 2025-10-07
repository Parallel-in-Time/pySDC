import numpy as np
from pySDC.projects.GPU.etc.generate_jobscript import write_jobscript


class LargeSim:
    config = None
    params = {}
    path = '/p/scratch/ccstma/baumann7/large_runs'
    email = 't.baumann@fz-juelich.de'

    def setup_CPU_params(self):
        raise NotImplementedError()

    def setup_GPU_params(self):
        raise NotImplementedError()

    @property
    def plotting_params(self):
        return {
            'tasks_per_node': 10,
            'partition': 'develgpus',
            'cluster': 'jusuf',
            'time': '3:60:00',
        }

    def write_jobscript_for_run(self, submit=True, restart_idx=0):
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
            '--mail-type=ALL',
            f'--mail-user={self.email}',
        ]
        if 'reservation' in self.params.keys():
            sbatch_options += [f'--reservation={self.params["reservation"]}']

        srun_options = [f'--tasks-per-node={tasks_per_node}']

        if useGPU:
            srun_options += ['--cpus-per-task=4', '--gpus-per-task=1']
            sbatch_options += ['--cpus-per-task=4', '--gpus-per-task=1']

        procs = (''.join(f'{me}/' for me in procs))[:-1]
        command = f'run_experiment.py --mode=run --res={res} --config={self.config} --procs={procs} -o {self.path} --restart_idx {restart_idx}'

        if useGPU:
            command += ' --useGPU=True'

        write_jobscript(sbatch_options, srun_options, command, cluster, submit=submit, name=f'{self.config}')

    def write_jobscript_for_plotting(self, num_procs=20, mode='plot', submit=True, restart_idx=0):
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
        command = f'run_experiment.py --mode={mode} --res={res} --config={self.config} --procs={procs_sim} -o {self.path} --restart_idx {restart_idx}'

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
            'procs': [1, 1, 4],
            'useGPU': False,
            'tasks_per_node': 16,
            'partition': 'batch',
            'cluster': 'jusuf',
            'res': 64,
            'time': '0:15:00',
        }

    def setup_GPU_params_test(self):
        """
        Test params with a small run.
        """
        self.params = {
            'procs': [1, 4, 1],
            'useGPU': True,
            'tasks_per_node': 4,
            'partition': 'develbooster',
            'cluster': 'booster',
            'res': 512,
            'time': '0:20:00',
        }

    def setup_GPU_params(self):
        """
        Params we actually want to use for the large simulation
        """
        self.params = {
            'procs': [1, 4, 192],
            'useGPU': True,
            'tasks_per_node': 4,
            'partition': 'booster',
            'cluster': 'booster',
            'res': 2304,
            'time': '1:00:00',
        }

    def setup_GPU_paramsFull(self):
        """
        Config for a stupid run on the whole machine.
        """
        self.params = {
            'procs': [1, 4, 896],
            'useGPU': True,
            'tasks_per_node': 4,
            'partition': 'largebooster',
            'cluster': 'booster',
            'res': 2688,
            'time': '0:20:00',
            'reservation': 'big-days-20241119',
        }


class RBCLarge(LargeSim):
    config = 'RBC_large'

    def setup_CPU_params(self):
        """
        Params for the large run
        """
        self.params = {
            'procs': [1, 4, 1024],
            'useGPU': False,
            'tasks_per_node': 128,
            'partition': 'dc-cpu',
            'cluster': 'jureca',
            'res': 4096,
            'time': '4:30:00',
        }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, default='GS', choices=['RBC', 'GS'])
    parser.add_argument('--XPU', type=str, choices=['CPU', 'GPU'], default='CPU')
    parser.add_argument('--mode', type=str, choices=['run', 'plot', 'video', 'plot_series'], default='plot')
    parser.add_argument('--submit', type=str, choices=['yes', 'no'], default='yes')
    parser.add_argument('--num_procs', type=int, default=10)
    parser.add_argument('--restart_idx', type=int, default=0)
    args = parser.parse_args()

    if args.problem == 'GS':
        cls = GSLarge
    elif args.problem == 'RBC':
        cls = RBCLarge
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
        sim.write_jobscript_for_run(submit=args.submit, restart_idx=args.restart_idx)
    elif args.mode == 'plot':
        sim.write_jobscript_for_plotting(num_procs=args.num_procs, submit=args.submit, restart_idx=args.restart_idx)
    elif args.mode == 'plot_series':
        sim.write_jobscript_for_plotting(num_procs=1, submit=args.submit, mode='plot_series')
    elif args.mode == 'video':
        sim.write_jobscript_for_video(num_procs=args.num_procs, submit=args.submit)
    else:
        raise NotImplementedError()
