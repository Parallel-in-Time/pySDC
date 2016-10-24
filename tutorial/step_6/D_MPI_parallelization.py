import subprocess
import os
import numpy as np
import filecmp

def main(cwd):

    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../..:.'

    # set lit of number of parallel steps
    num_procs_list = [2**i for i in range(int(np.log2(16)+1))]

    # set up new/empty file for output
    f = open(cwd+'/step_6_D_out.txt', 'w')
    f.close()

    # run code with different number of MPI processes
    for num_procs in num_procs_list:
        print('Running code with %2i processes...' %num_procs)
        cmd = ('mpirun -np ' + str(num_procs) + ' python playground_parallelization.py').split()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr = subprocess.PIPE, env=my_env, cwd=cwd)
        p.wait()
        assert p.returncode == 0, 'ERROR: did not get return code 0, got %s' % p.returncode

if __name__ == "__main__":
    main('.')
