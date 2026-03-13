def parse_args():
    import argparse

    def cast_to_bool(me):
        return False if me in ['False', '0', 0] else True

    def str_to_procs(me):
        procs = me.split('/')
        assert len(procs) == 3
        return [int(p) for p in procs]

    parser = argparse.ArgumentParser()
    parser.add_argument('--useGPU', type=cast_to_bool, help='Toggle for GPUs', default=False)
    parser.add_argument(
        '--mode',
        type=str,
        help='Mode for this script',
        default=None,
        choices=['run', 'plot', 'benchmark'],
    )
    parser.add_argument('--config', type=str, help='Configuration to load', default=None)
    parser.add_argument('--restart_idx', type=int, help='Restart from file by index', default=0)
    parser.add_argument('--procs', type=str_to_procs, help='Processes in steps/sweeper/space', default='1/1/1')
    parser.add_argument('--res', type=int, help='Space resolution along first axis', default=-1)
    parser.add_argument('--dt', type=float, help='(Starting) Step size', default=-1)
    parser.add_argument(
        '--logger_level', type=int, help='Logger level on the first rank in space and in the sweeper', default=15
    )
    parser.add_argument('-o', type=str, help='output path', default='./')
    parser.add_argument(
        '--distribution',
        type=str,
        help='distribute tasks',
        default='space_first',
        choices=['space_first', 'time_first'],
    )

    return vars(parser.parse_args())


def run_experiment(args, config, **kwargs):
    import pickle
    import os

    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.helpers.stats_helper import filter_stats

    type(config).base_path = args['o']
    os.makedirs(f'{args["o"]}/data', exist_ok=True)

    if args['mode'] == 'benchmark':
        config.prepare_for_benchmark()

    description = config.get_description(
        useGPU=args['useGPU'], MPIsweeper=args['procs'][1] > 1, res=args['res'], dt=args['dt'], **kwargs
    )
    controller_params = config.get_controller_params(logger_level=args['logger_level'])

    if args['mode'] == 'benchmark':
        config.prepare_description_for_benchmark(description, controller_params)

    if args['useGPU']:
        from pySDC.implementations.hooks.log_timings import GPUTimings

        controller_params['hook_class'].append(GPUTimings)

    assert (
        config.comms[0].size == 1
    ), 'Have not figured out how to do MPI controller with GPUs yet because I need NCCL for that!'
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
    prob = controller.MS[0].levels[0].prob

    u0, t0 = config.get_initial_condition(prob, restart_idx=args['restart_idx'])

    if args['mode'] == 'benchmark':
        config.prepare_caches_for_benchmark(prob, controller)

    config.prepare_caches(prob)

    uend, stats = controller.run(u0=u0, t0=t0, Tend=config.Tend)

    combined_stats = filter_stats(stats, comm=config.comm_world)

    if config.comm_world.rank == config.comm_world.size - 1:
        path = f'{config.base_path}/data/{config.get_path()}-stats-whole-run.pickle'
        with open(path, 'wb') as file:
            pickle.dump(combined_stats, file)
        print(f'Stored stats in {path}', flush=True)

    return uend


if __name__ == '__main__':
    from pySDC.projects.RayleighBenard.RBC3D_configs import get_config

    args = parse_args()

    config = get_config(args)

    if args['mode'] in ['run', 'benchmark']:
        run_experiment(args, config)
    else:
        raise NotImplementedError
