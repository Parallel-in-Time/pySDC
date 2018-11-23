
import indiesolver
import numpy as np
import scipy

from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.collocation_classes.gauss_lobatto import CollGaussLobatto
from pySDC.implementations.collocation_classes.equidistant_right import EquidistantNoLeft


def evaluate(solution):
    x = solution["parameters"]

    mf = 3

    collf = CollGaussRadau_Right(num_nodes=mf, tleft=0.0, tright=1.0)
    # coll = CollGaussLobatto(num_nodes=m, tleft=0.0, tright=1.0)
    # coll = EquidistantNoLeft(num_nodes=m, tleft=0.0, tright=1.0)

    Qf = collf.Qmat[1:, 1:]
    Idf = np.eye(mf)

    QT = Qf.T
    [_, _, U] = scipy.linalg.lu(QT, overwrite_a=True)
    Qdf = U.T

    mc = int((mf + 1) / 2)

    collc = CollGaussRadau_Right(num_nodes=mc, tleft=0.0, tright=1.0)
    # coll = CollGaussLobatto(num_nodes=m, tleft=0.0, tright=1.0)
    # coll = EquidistantNoLeft(num_nodes=m, tleft=0.0, tright=1.0)

    Qc = collc.Qmat[1:, 1:]
    Idc = np.eye(mc)

    QT = Qc.T
    [_, _, U] = scipy.linalg.lu(QT, overwrite_a=True)
    Qdc = U.T

    sum1r = x['x11r'] + x['x12r'] + x['x13r']
    sum2r = x['x21r'] + x['x22r'] + x['x23r']
    sum1i = x['x11i'] + x['x12i']
    sum2i = x['x21i'] + x['x22i']
    sum3i = x['x31i'] + x['x32i']

    if sum1r == 0.0 or sum2r == 0.0 or sum1i == 0.0 or sum2i == 0.0 or sum3i == 0.0:
        solution["metrics"] = {}
        solution["metrics"]["rho"] = 99
        return solution

    Tr = np.array([[x['x11r'] / sum1r, x['x12r'] / sum1r, x['x13r'] / sum1r],
                   [x['x21r'] / sum2r, x['x22r'] / sum2r, x['x23r'] / sum2r]])
    Ti = np.array([[x['x11i'] / sum1i, x['x12i'] / sum1i],
                   [x['x21i'] / sum2i, x['x22i'] / sum2i],
                   [x['x31i'] / sum3i, x['x32i'] / sum3i]])

    # THIS WORKS REALLY WELL! No need to take imaginary parts in x, though (found minimum has zero imaginary parts)
    k = 0
    obj_val = 0.0
    for i in range(-8, 8):
        for l in range(-8, 8):
            k += 1
            lamdt = -10 ** i + 1j * 10 ** l
            C = Idf - lamdt * Qf
            Pf = Idf - lamdt * Qdf
            Rf = Idf - np.linalg.inv(Pf).dot(C)
            Pc = Idc - lamdt * Qdc
            Rc = Idf - Ti.dot(np.linalg.inv(Pc)).dot(Tr).dot(C)
            R = Rf.dot(Rc)
            rhoR = max(abs(np.linalg.eigvals(R)))
            obj_val += rhoR

    obj_val /= k

    solution["metrics"] = {}
    solution["metrics"]["rho"] = obj_val
    return solution


# y = [5.8054876, 8.46779587, 17.72188108, 6.75505219, 5.53129906]
y = [0.0, 0.0, 0.0, 0.0, 0.0]
# y = [1.0, 1.0, 1.0, 1.0, 1.0]
ymax = 20.0
ymin = -20.0
params = dict()
params['x11r'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[0]}
params['x12r'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[1]}
params['x13r'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[2]}
params['x21r'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[3]}
params['x22r'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[4]}
params['x23r'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[0]}
params['x11i'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[1]}
params['x12i'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[2]}
params['x21i'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[3]}
params['x22i'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[4]}
params['x31i'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[3]}
params['x32i'] = {'type': 'float', 'space': 'decision', 'min': ymin, 'max': ymax, 'init': y[4]}

problem = {'problem_name': 'Qdelta_sum_ml',
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
for iteration in range(0, 12500):
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
            if curr_min == rho or solution["ID"] % 1000 == 0:
                print(solution["ID"], curr_min, pars)
        worker.tell_metrics(solutions)
    else:
        print(reply)
        exit()
