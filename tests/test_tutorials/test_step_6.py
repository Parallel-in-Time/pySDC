import subprocess, os

from tutorial.step_6.A_visualize_residuals import main as main_A
from tutorial.step_6.B_classic_vs_multigrid_controller import main as main_B
from tutorial.step_6.C_multistep_SDC import main as main_C

# def test_A():
#     main_A()
#
# def test_B():
#     main_B()
# #
# def test_C():
#     main_C()

def test_MPI():
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../..:.'
    cmd = 'mpirun -np 3 python examples/heat1d/playground_parallel.py'.split()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr = subprocess.PIPE, env=my_env)
    out, err = p.communicate()
    # print(out)
    # print(err)
    assert not err
    assert float(out.split()[-1]) < 7.823e-07

