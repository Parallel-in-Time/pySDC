PROJECT_PATH = '/p/project1/ccstma/baumann7/pySDC/pySDC/projects/GPU'
DEFAULT_SBATCH_OPTIONS = ['-A cstma', '--threads-per-core=1', f'--output={PROJECT_PATH}/etc/slurm-out/%j.txt']
DEFAULT_SRUN_OPTIONS = ['--cpu-bind=sockets']


def generate_directories():
    '''
    Initialize directories for jobscripts and slurm output
    '''
    import os

    for name in ['jobscripts', 'slurm-out']:
        path = f'{PROJECT_PATH}/etc/{name}'
        os.makedirs(path, exist_ok=True)


def get_jobscript_text(sbatch_options, srun_options, command, cluster):
    """
    Generate the text for a jobscript

    Args:
        sbatch_options (list): List of options for sbatch
        srun_options (list): Options for the srun command
        command (str): python (!) command. Will be prefaced by `python <path>/`
        cluster (str): Name of the cluster you want to run on

    Returns:
        str: Content of jobscript
    """
    msg = '#!/usr/bin/bash\n\n'
    for op in DEFAULT_SBATCH_OPTIONS + sbatch_options:
        msg += f'#SBATCH {op}\n'

    msg += f'\nsource {PROJECT_PATH}/etc/venv_{cluster.lower()}/activate.sh\n'

    srun_cmd = 'srun'
    for op in DEFAULT_SRUN_OPTIONS + srun_options:
        srun_cmd += f' {op}'

    msg += f'\n{srun_cmd} python {PROJECT_PATH}/{command}'
    return msg


def write_jobscript(sbatch_options, srun_options, command, cluster, submit=True):
    """
    Generate a jobscript.

    Args:
        sbatch_options (list): List of options for sbatch
        srun_options (list): Options for the srun command
        command (str): python (!) command. Will be prefaced by `python <path>/`
        cluster (str): Name of the cluster you want to run on
        submit (bool): If yes, the script will be submitted to SLURM after it is written
    """
    generate_directories()

    text = get_jobscript_text(sbatch_options, srun_options, command, cluster)

    path = f'{PROJECT_PATH}/etc/jobscripts/{command.replace(" ", "").replace("/", "_")}-{cluster}.sh'
    with open(path, 'w') as file:
        file.write(text)

    if submit:
        import os

        os.system(f'sbatch {path}')
