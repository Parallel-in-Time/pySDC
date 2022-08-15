import indiesolver
import numpy as np


def evaluate(solution):
    x = solution["parameters"]

    m = 5

    # coll = CollGaussRadau_Right(num_nodes=m, tleft=0.0, tright=1.0)
    coll = CollGaussLobatto(num_nodes=m, tleft=0.0, tright=1.0)
    # coll = EquidistantNoLeft(num_nodes=m, tleft=0.0, tright=1.0)
    # print(coll.nodes)
    # exit()

    QQ = np.zeros(np.shape(coll.Qmat))

    # if we have Gauss-Lobatto nodes, we can do a magic trick from the Book
    # this takes Gauss-Lobatto IIIB and create IIIA out of this
    if isinstance(coll, CollGaussLobatto):

        for i in range(coll.num_nodes):
            for j in range(coll.num_nodes):
                QQ[i + 1, j + 1] = coll.weights[j] * (1.0 - coll.Qmat[j + 1, i + 1] / coll.weights[i])
        QQ = np.dot(coll.Qmat, QQ)

    # if we do not have Gauss-Lobatto, just multiply Q (will not get a symplectic method, they say)
    else:
        exit()
        QQ = np.dot(coll.Qmat, coll.Qmat)

    QQ = QQ[1:, 1:]

    Qd = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [x['x21'], 0.0, 0.0, 0.0, 0.0],
            [x['x31'], x['x32'], 0.0, 0.0, 0.0],
            [x['x41'], x['x42'], x['x43'], 0.0, 0.0],
            [x['x51'], x['x52'], x['x53'], x['x54'], 0.0],
        ]
    )

    # THIS WORKS REALLY WELL! No need to take imaginary parts in x, though (found minimum has zero imaginary parts)
    k = 0
    obj_val = 0.0
    for i in range(-8, 4):
        k += 1
        lamdt = -(10**i)
        try:
            R = lamdt * np.linalg.inv(np.eye(m) - lamdt * Qd).dot(QQ - Qd)
        except np.linalg.linalg.LinAlgError:
            obj_val += 99
            continue

        rhoR = max(abs(np.linalg.eigvals(R)))
        obj_val += rhoR

    obj_val /= k

    solution["metrics"] = {}
    solution["metrics"]["rho"] = obj_val
    return solution


# y = [5.8054876, 8.46779587, 17.72188108, 6.75505219, 5.53129906]
# y = [0.0, 0.0, 0.0, 0.0, 0.0]
y = [1.0, 1.0, 1.0, 1.0, 1.0]
ymax = 20.0
ymin = -20.0
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
# params['x5'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[4]}
# params['x6'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[3]}
# params['x7'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[4]}
# params['x8'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[3]}
# params['x9'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[4]}

params['x21'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x31'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x32'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x41'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x42'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x43'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x51'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x52'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x53'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}
params['x54'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': 0.0}

problem = {
    'problem_name': 'Qdelta_sum_verlet_ld',
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
for iteration in range(0, 25000):
    reply = worker.ask_new_solutions(4)
    solutions = dict()
    solutions["solutions"] = []
    if reply["status"] == "success":
        for solution in reply["solutions"]:
            solutions["solutions"].append(evaluate(solution))
            rho = solution["metrics"]["rho"]
            curr_min = min(curr_min, rho)
            if curr_min == rho:
                pars = [solution["parameters"][k] for k in solution["parameters"]]
            if curr_min == rho or solution["ID"] % 100 == 0:
                print(solution["ID"], curr_min, pars)
        worker.tell_metrics(solutions)
    else:
        print(reply)
        exit()
