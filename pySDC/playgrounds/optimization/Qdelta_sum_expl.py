import indiesolver
import numpy as np


def evaluate(solution):
    x = solution["parameters"]

    m = 5

    coll = CollGaussRadau_Right(num_nodes=m, tleft=0.0, tright=1.0)
    Q = coll.Qmat[1:, 1:]

    var = [x['x' + str(j)] for j in range(1, m)]
    # var = [x['x' + str(j) + 'r'] + 1j * x['x' + str(j) + 'i'] for j in range(1, m + 1)]

    Qd = np.zeros((m, m))
    Qd[1, 0] = var[0]
    Qd[2, 1] = var[1]
    Qd[3, 2] = var[2]
    Qd[4, 3] = var[3]

    # THIS WORKS REALLY WELL! No need to take imaginary parts in x, though (found minimum has zero imaginary parts)
    k = 0
    obj_val = 0.0
    for i in range(-8, 1):
        for l in range(-8, 1):
            k += 1
            lamdt = -(10**i) + 1j * 10**l
            R = lamdt * np.linalg.inv(np.eye(m) - lamdt * Qd).dot(Q - Qd)
            rhoR = max(abs(np.linalg.eigvals(R)))
            obj_val += rhoR

    obj_val /= k

    solution["metrics"] = {}
    solution["metrics"]["rho"] = obj_val
    return solution


y = [0.0, 0.0, 0.0, 0.0]
# y = [1.0, 1.0, 1.0, 1.0]
ymax = 20.0
ymin = 0.0
params = dict()
# params['x1r'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[0]}
# params['x2r'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[1]}
# params['x3r'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[2]}
# params['x4r'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[3]}
# params['x5r'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[4]}
# params['x1i'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[0]}
# params['x2i'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[1]}
# params['x3i'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[2]}
# params['x4i'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[3]}
# params['x5i'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[4]}

params['x1'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[0]}
params['x2'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[1]}
params['x3'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[2]}
params['x4'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[3]}

problem = {
    'problem_name': 'Qdelta_sum_expl',
    'parameters': params,
    'metrics': {'rho': {'type': 'objective', 'goal': 'minimize'}},
}

worker = indiesolver.indiesolver()
worker.initialize("indiesolver.com", 8080, "dg8f5a0dd9ed")
reply = worker.create_problem(problem)

if reply["status"] != "success":
    print(reply)
    exit()

curr_min = 99
pars = None
for iteration in range(0, 100000):
    reply = worker.ask_new_solutions(8)
    solutions = dict()
    solutions["solutions"] = []
    if reply["status"] == "success":
        for solution in reply["solutions"]:
            solutions["solutions"].append(evaluate(solution))
            rho = solution["metrics"]["rho"]
            curr_min = min(curr_min, rho)
            if curr_min == rho:
                pars = [solution["parameters"][k] for k in solution["parameters"]]
                print(solution["ID"], curr_min, pars)
        worker.tell_metrics(solutions)
    else:
        print(reply)
        exit()
