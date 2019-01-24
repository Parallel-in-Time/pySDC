
import indiesolver
import numpy as np

from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right


def evaluate(solution):
    x = solution["parameters"]

    m = 5

    coll = CollGaussRadau_Right(num_nodes=m, tleft=0.0, tright=1.0)
    Q = coll.Qmat[1:, 1:]

    Qd = np.array([[x['x11'], x['x12'], x['x13'], x['x14'], x['x15']],
                   [0.0, x['x22'], x['x23'], x['x24'], x['x25']],
                   [0.0, 0.0, x['x33'], x['x34'], x['x35']],
                   [0.0, 0.0, 0.0, x['x44'], x['x45']],
                   [0.0, 0.0, 0.0, 0.0, x['x55']]])

    nsteps = 8

    E = np.zeros((nsteps, nsteps))
    np.fill_diagonal(E[1:, :], 1)

    Ea = E.copy()
    Ea[0, -1] = 0.001
    # Da, Va = np.linalg.eig(Ea)
    # Via = np.linalg.inv(Va)

    N = np.zeros((m, m))
    N[:, -1] = 1

    k = 0
    obj_val = 0.0
    for i in range(-8, 8):
        for l in range(-8, 8):
            k += 1
            lamdt = -10 ** i + 1j * 10 ** l
            R = np.linalg.inv(np.eye(nsteps * m) - lamdt * np.kron(np.eye(nsteps), Qd) - np.kron(Ea, N)).dot(
                lamdt * np.kron(np.eye(nsteps), Q - Qd) + np.kron(E - Ea, N))
            rhoR = max(abs(np.linalg.eigvals(R)))
            obj_val += rhoR

    obj_val /= k

    solution["metrics"] = {}
    solution["metrics"]["rho"] = obj_val
    return solution


# y = [5.8054876, 8.46779587, 17.72188108, 6.75505219, 5.53129906]
# y = [1.0, 1.0, 1.0, 1.0, 1.0]
y = [0.0, 0.0, 0.0, 0.0, 0.0]
ymax = 20.0
ymin = -20.0
params = dict()
params['x11'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[0]}
params['x12'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x13'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[1]}
params['x14'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x15'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x22'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[2]}
params['x23'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x24'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x25'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x33'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[3]}
params['x34'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x35'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x44'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x45'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x55'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[4]}

problem = {'problem_name': 'Qdelta_ud_sum_diag',
           'parameters': params,
           'metrics': {'rho': {'type': 'objective', 'goal': 'minimize'}}}

worker = indiesolver.indiesolver()
worker.initialize("indiesolver.com", 8080, "dg8f5a0dd9ed")
reply = worker.create_problem(problem)

if reply["status"] != "success":
    print(reply)
    exit()

curr_min = 99
pars = None
for iteration in range(0, 1):
    reply = worker.ask_new_solutions(1)
    solutions = dict()
    solutions["solutions"] = []
    if reply["status"] == "success":
        for solution in reply["solutions"]:
            solutions["solutions"].append(evaluate(solution))
            rho = solution["metrics"]["rho"]
            curr_min = min(curr_min, rho)
            if curr_min == rho:
                pars = [solution["parameters"][k] for k in solution["parameters"]]
            if curr_min == rho or solution["ID"] % 1000 == 0:
                print(solution["ID"], curr_min, pars)
        worker.tell_metrics(solutions)
    else:
        print(reply)
        exit()
