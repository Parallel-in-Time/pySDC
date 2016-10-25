import subprocess
import os
import numpy as np
import filecmp

def main(cwd):

    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../..:.'

    # set lit of number of parallel steps
    num_procs_list = [1,2,4,8]

    # set up new/empty file for output
    fname = 'step_6_C1_out.txt'
    f = open(cwd + '/../../' + fname, 'w')
    f.close()

    # run code with different number of MPI processes
    for num_procs in num_procs_list:
        print('Running code with %2i processes...' %num_procs)
        cmd = ('mpirun -np ' + str(num_procs) + ' python playground_parallelization.py ../../' + fname).split()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr = subprocess.PIPE, env=my_env, cwd=cwd)
        p.wait()
        assert p.returncode == 0, 'ERROR: did not get return code 0, got %s' % p.returncode

    # set lit of number of parallel steps
    num_procs_list = [3,5,7,9]

    # set up new/empty file for output
    fname = 'step_6_C2_out.txt'
    f = open(cwd + '/../../' + fname, 'w')
    f.close()

    # run code with different number of MPI processes
    for num_procs in num_procs_list:
        print('Running code with %2i processes...' % num_procs)
        cmd = ('mpirun -np ' + str(num_procs) + ' python playground_parallelization.py ../../' + fname).split()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env, cwd=cwd)
        p.wait()
        assert p.returncode == 0, 'ERROR: did not get return code 0, got %s' % p.returncode


if __name__ == "__main__":
    main('.')
