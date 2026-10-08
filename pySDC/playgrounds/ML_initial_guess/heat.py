import torch
from pySDC.playgrounds.ML_initial_guess.sweeper import GenericImplicitML_IG
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.tutorial.step_7.torch_heat import Heat1DFDTensor


def main():
    """
    A simple test program to setup a full step instance
    """
    dt = 1e-2

    level_params = dict()
    level_params['restol'] = 1e-10
    level_params['dt'] = dt

    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'
    sweeper_params['initial_guess'] = 'NN'

    problem_params = dict()

    step_params = dict()
    step_params['maxiter'] = 20

    description = dict()
    description['problem_class'] = Heat1DFDTensor
    description['problem_params'] = problem_params
    description['sweeper_class'] = GenericImplicitML_IG
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 20}, description=description)

    P = controller.MS[0].levels[0].prob

    uinit = P.u_exact(0)
    uend, _ = controller.run(u0=uinit, t0=0, Tend=dt)
    u_exact = P.u_exact(dt)
    print("error ", torch.abs(u_exact - uend).max())


if __name__ == "__main__":
    main()
