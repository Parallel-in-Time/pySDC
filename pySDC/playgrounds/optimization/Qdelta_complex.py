
import indiesolver
import numpy as np

from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right


def evaluate(solution):
    x = solution["parameters"]

    m = 5
    coll = CollGaussRadau_Right(num_nodes=m, tleft=0.0, tright=1.0)
    Q = coll.Qmat[1:, 1:]

    var = [x['x' + str(j) + 'r'] + 1j * x['x' + str(j) + 'i'] for j in range(1, m + 1)]

    invQd = np.diag(var)

    R = np.eye(m) - invQd.dot(Q)
    rhoR = max(abs(np.linalg.eigvals(R)))
    obj_val = rhoR

    solution["metrics"] = {}
    solution["metrics"]["rho"] = obj_val
    return solution


# y = [5.8054876, 8.46779587, 17.72188108, 6.75505219, 5.53129906]
y = [1.0, 1.0, 1.0, 1.0, 1.0]
ymax = 20.0
ymin = 0.0
params = {}
params['x1r'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[0]}
params['x2r'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[1]}
params['x3r'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[2]}
params['x4r'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[3]}
params['x5r'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[4]}
params['x1i'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[0]}
params['x2i'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[1]}
params['x3i'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[2]}
params['x4i'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[3]}
params['x5i'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[4]}

problem = {'problem_name': 'Qdelta_complex',
           'parameters': params,
           'metrics': { 'rho' : { 'type': 'objective', 'goal': 'minimize'}}}

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
    solutions = {}
    solutions["solutions"] = []
    if (reply["status"] == "success"):
        for solution in reply["solutions"]:
            solutions["solutions"].append( evaluate(solution) )
        worker.tell_metrics(solutions)
        rho = reply["solutions"][0]["metrics"]["rho"]
        curr_min = min(curr_min, rho)
        if curr_min == rho:
            pars = reply["solutions"][0]["parameters"]
        print(iteration, rho, curr_min, pars)
    else:
        print(reply)
        exit()

