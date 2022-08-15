import indiesolver
import numpy as np
import scipy


def evaluate(solution):
    x = solution["parameters"]

    m = 3

    coll = CollGaussRadau_Right(num_nodes=m, tleft=0.0, tright=1.0)
    # coll = CollGaussLobatto(num_nodes=m, tleft=0.0, tright=1.0)
    # coll = EquidistantNoLeft(num_nodes=m, tleft=0.0, tright=1.0)
    # print(coll.nodes)
    # exit()
    Q = coll.Qmat[1:, 1:]

    var = [x['x' + str(j)] for j in range(1, m + 1)]
    # var = [x['x' + str(j) + 'r'] + 1j * x['x' + str(j) + 'i'] for j in range(1, m + 1)]

    Qd = np.diag(var)

    QT = Q.T
    [_, _, U] = scipy.linalg.lu(QT, overwrite_a=True)
    Qd_LU = U.T

    # THIS WORKS REALLY WELL! No need to take imaginary parts in x, though (found minimum has zero imaginary parts)
    k = 0
    obj_val = 0.0
    for i in range(-8, 8):
        for l in range(-8, 8):
            k += 1
            lamdt = -(10**i) + 1j * 10**l
            R = lamdt * np.linalg.inv(np.eye(m) - lamdt * Qd).dot(Q - Qd)
            R_LU = lamdt * np.linalg.inv(np.eye(m) - lamdt * Qd_LU).dot(Q - Qd_LU)
            rhoR = max(abs(np.linalg.eigvals(R_LU.dot(R))))
            obj_val += rhoR

    obj_val /= k

    solution["metrics"] = {}
    solution["metrics"]["rho"] = obj_val
    return solution


# y = [5.8054876, 8.46779587, 17.72188108, 6.75505219, 5.53129906]
y = [0.0, 0.0, 0.0, 0.0, 0.0]
# y = [1.0, 1.0, 1.0, 1.0, 1.0]
ymax = 10000.0
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
# params['x4'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[3]}
# params['x5'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[4]}
# params['x6'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[3]}
# params['x7'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[4]}
# params['x8'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[3]}
# params['x9'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[4]}

problem = {
    'problem_name': 'Qdelta_sum_double',
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
