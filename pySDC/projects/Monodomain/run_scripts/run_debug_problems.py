from pathlib import Path
import time
import numpy as np

from pySDC.core.Errors import ParameterError

from pySDC.projects.ExplicitStabilized.problem_classes.debug_problems import debug_exp_runge_kutta_1, debug_exp_runge_kutta_2

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.projects.ExplicitStabilized.sweeper_classes.runge_kutta.imexexp_1st_order import imexexp_1st_order
from pySDC.projects.ExplicitStabilized.sweeper_classes.exponential_runge_kutta.imexexp_1st_order import imexexp_1st_order as imexexp_1st_order_ExpRK
from pySDC.projects.ExplicitStabilized.sweeper_classes.runge_kutta.exponential_splitting_explicit_stabilized import exponential_splitting_explicit_stabilized
from pySDC.projects.ExplicitStabilized.sweeper_classes.runge_kutta.explicit_stabilized import explicit_stabilized
from pySDC.projects.ExplicitStabilized.sweeper_classes.runge_kutta.multirate_explicit_stabilized import multirate_explicit_stabilized
from pySDC.projects.ExplicitStabilized.sweeper_classes.runge_kutta.exponential_multirate_explicit_stabilized import exponential_multirate_explicit_stabilized
from pySDC.projects.ExplicitStabilized.sweeper_classes.runge_kutta.splitting_explicit_stabilized import splitting_explicit_stabilized

from pySDC.projects.ExplicitStabilized.explicit_stabilized_classes.es_methods import RKW1, RKC1, RKU1, HSRKU1, mRKC1


def main():
    # define integration methods
    integrators = ["IMEX"]
    integrators = ["IMEXEXP"]
    integrators = ["IMEXEXP_EXPRK"]
    # integrators = ['ES']
    # integrators = ['mES']
    # integrators = ['exp_mES']
    # integrators = ['split_ES']
    # integrators = ['exp_split_ES']

    num_procs = 1

    ref = 0

    # initialize level parameters
    level_params = dict()
    level_params["restol"] = 5e-8
    level_params["dt"] = 0.1 / 2**ref
    level_params["nsweeps"] = [3]
    level_params["residual_type"] = "full_rel"

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params["initial_guess"] = "spread"
    sweeper_params["quad_type"] = "RADAU-RIGHT"
    sweeper_params["num_nodes"] = [3]
    # specific for explicit stabilized methods
    sweeper_params["es_class"] = RKW1
    sweeper_params["es_class_outer"] = RKW1
    sweeper_params["es_class_inner"] = RKW1
    # sweeper_params['es_s_outer'] = 0 # if given, or not zero, then the algorithm fixes s of the outer stabilized scheme to this value.
    # sweeper_params['es_s_inner'] = 0
    # sweeper_params['res_comp'] = 'f_eta'
    sweeper_params["damping"] = 0.05
    sweeper_params["safe_add"] = 0
    # sweeper_params['order'] = [3]
    # sweeper_params['nodes_choice'] = 'all' # closest_radau, last, all
    sweeper_params["rho_freq"] = 100

    # initialize step parameters
    step_params = dict()
    step_params["maxiter"] = 1

    # initialize problem parameters
    problem_params = dict()

    # base transfer parameters
    base_transfer_params = dict()
    base_transfer_params["finter"] = False

    # initialize controller parameters
    controller_params = dict()
    controller_params["predict_type"] = "pfasst_burnin"
    controller_params["log_to_file"] = False
    controller_params["fname"] = "data/ExplicitStabilized_HeatEquation"
    controller_params["logger_level"] = 20
    controller_params["dump_setup"] = False

    Path("data").mkdir(parents=True, exist_ok=True)

    for integrator in integrators:
        description = dict()
        if integrator == "IMEX":
            description["sweeper_class"] = imex_1st_order
        elif integrator == "IMEXEXP":
            description["sweeper_class"] = imexexp_1st_order
        elif integrator == "IMEXEXP_EXPRK":
            description["sweeper_class"] = imexexp_1st_order_ExpRK
        elif integrator == "ES":
            description["sweeper_class"] = explicit_stabilized
        elif integrator == "mES":
            description["sweeper_class"] = multirate_explicit_stabilized
        elif integrator == "exp_mES":
            description["sweeper_class"] = exponential_multirate_explicit_stabilized
        elif integrator == "split_ES":
            description["sweeper_class"] = splitting_explicit_stabilized
        elif integrator == "exp_split_ES":
            description["sweeper_class"] = exponential_splitting_explicit_stabilized
        else:
            raise ParameterError("Unknown integrator.")

        description["problem_class"] = debug_exp_runge_kutta_2
        description["problem_params"] = problem_params
        description["sweeper_params"] = sweeper_params
        description["level_params"] = level_params
        description["step_params"] = step_params
        description["base_transfer_params"] = base_transfer_params

        # instantiate the controller
        controller = controller_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)

        # get initial values on finest level
        P = controller.MS[0].levels[0].prob
        # set time parameters
        t0 = P.t0
        Tend = P.Tend
        uinit = P.initial_value()

        prob_size = P.get_size()
        print(f"Problem size: {prob_size}")

        # call main function to get things done...
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

        # compute exact solution and compare
        if P.know_exact:
            P.compute_errors(uend, Tend)

        # filter statistics by type (number of iterations)
        iter_counts = get_sorted(stats, type="niter", sortby="time")

        niters = np.array([item[1] for item in iter_counts])
        out = "Mean number of iterations: %4.2f" % np.mean(niters)
        print(out)
        out = "Std and var for number of iterations: %4.2f -- %4.2f" % (float(np.std(niters)), float(np.var(niters)))
        print(out)

        timing = get_sorted(stats, type="timing_run", sortby="time")
        out = f"Time to solution: {timing[0][1]:6.4f} sec."
        print(out)


if __name__ == "__main__":
    main()
