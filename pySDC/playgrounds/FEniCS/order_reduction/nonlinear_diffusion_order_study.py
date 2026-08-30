"""Run the three-way nonlinear-diffusion order-reduction experiment."""

import json
from pathlib import Path

import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.playgrounds.FEniCS.order_reduction.nonlinear_diffusion import (
    manufactured_nonlinear_diffusion_cosine,
    manufactured_nonlinear_diffusion_cosine_lift,
    manufactured_nonlinear_diffusion_sine,
)


def run_sdc(problem_class, dt, num_nodes=3, Tend=1.0, nvars=255, **kwargs):
    description = {
        'problem_class': problem_class,
        'problem_params': {'nvars': nvars, **kwargs},
        'sweeper_class': generic_implicit,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': num_nodes},
        'level_params': {'restol': 1e-12, 'dt': dt},
        'step_params': {'maxiter': 100},
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    P = controller.MS[0].levels[0].prob
    lifted = isinstance(P, manufactured_nonlinear_diffusion_cosine_lift)
    initial = P.u_exact_lifted(0.0) if lifted else P.u_exact(0.0)
    result, _ = controller.run(u0=initial, t0=0.0, Tend=Tend)
    if lifted:
        result = result + P.lift(Tend)
    exact = P.u_exact(Tend)
    return float(abs(exact - result) / abs(exact))


def main():
    dts = [0.5 / 2**k for k in range(5)]
    common = {'Tend': 1.0, 'nvars': 63, 'gamma': 0.1, 'c': 0.25}
    definitions = [
        (manufactured_nonlinear_diffusion_sine, 'Nonlinear diffusion sine (constant-in-time BCs)'),
        (manufactured_nonlinear_diffusion_cosine, 'Nonlinear diffusion cosine (time-dependent BCs)'),
        (manufactured_nonlinear_diffusion_cosine_lift, 'Nonlinear diffusion cosine + boundary lifting'),
    ]
    cases = []
    for cls, label in definitions:
        errors = [run_sdc(cls, dt, **common) for dt in dts]
        errors_array = np.asarray(errors)
        dts_array = np.asarray(dts)
        local = (np.log(errors_array[:-1] / errors_array[1:]) / np.log(dts_array[:-1] / dts_array[1:])).tolist()
        fitted = float(np.polyfit(np.log(dts), np.log(errors), 1)[0])
        print(label, ['%.4e' % error for error in errors], 'local', [round(order, 3) for order in local])
        cases.append({'label': label, 'problem_class': cls.__name__, 'num_nodes': 3, 'expected_order': 5,
                      'dts': dts, 'errors': errors, 'local_orders': local, 'fitted_order': fitted})
    output = Path(__file__).parent / 'nonlinear_diffusion_order_study_results.json'
    output.write_text(json.dumps({'meta': {'equation': 'u_t = d_x((1 + gamma*u^2)*u_x) + f', **common, 'dts': dts}, 'cases': cases}, indent=2))
    print(f'Wrote results to {output}')


if __name__ == '__main__':
    main()
