
import indiesolver
import json
import numpy as np
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right


def evaluate(solution):
    x = solution["parameters"]

    m = 5

    coll = CollGaussRadau_Right(num_nodes=m, tleft=0.0, tright=1.0)
    Q = coll.Qmat[1:, 1:]

    Qd = np.array([[x['x11'], 0.0, 0.0, 0.0, 0.0],
                   [x['x21'], x['x22'], 0.0, 0.0, 0.0],
                   [x['x31'], x['x32'], x['x33'], 0.0, 0.0],
                   [x['x41'], x['x42'], x['x43'], x['x44'], 0.0],
                   [x['x51'], x['x52'], x['x53'], x['x54'], x['x55']]])

    k = 0
    obj_val = 0.0
    for i in range(-8, 8):
        for l in range(-8, 8):
            k += 1
            lamdt = -10 ** i + 1j * 10 ** l
            R = lamdt * np.linalg.inv(np.eye(m) - lamdt * Qd).dot(Q - Qd)
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
ymin = 0.0
params = dict()
params['x11'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[0]}
params['x21'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x22'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[1]}
params['x31'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x32'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x33'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[2]}
params['x41'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x42'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x43'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x44'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[3]}
params['x51'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x52'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x53'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x54'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x55'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[4]}

problem = {'problem_name': 'Qdelta_ld_sum',
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
for iteration in range(0, 100000):
    reply = worker.ask_new_solutions(1)
    solutions = dict()
    solutions["solutions"] = []
    if reply["status"] == "success":
        for solution in reply["solutions"]:
            solutions["solutions"].append(evaluate(solution))
        worker.tell_metrics(solutions)
        rho = reply["solutions"][0]["metrics"]["rho"]
        curr_min = min(curr_min, rho)
        if curr_min == rho:
            pars = reply["solutions"][0]["parameters"]
        print(iteration, rho, curr_min, pars)
    else:
        print(reply)
        exit()
