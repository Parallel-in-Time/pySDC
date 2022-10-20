import indiesolver


def evaluate(solution):
    x = solution["parameters"]
    obj_val = 0
    if x['x1'] == "A":
        obj_val = 0.2 + pow(x['x2'] - 0.3456789, 2.0)
    if x['x1'] == "B":
        obj_val = 0.1 + pow(x['x2'] - 0.3456789, 2.0) + pow(x['x3'] - 0.3456789, 2.0)
    if x['x1'] == "C":
        obj_val = pow(x['x2'] - 0.456789, 2.0) + pow(x['x3'] - x['x2'] - 0.234567, 2.0)

    obj_val = obj_val + pow(x['x4'] - 7, 2.0)

    solution["metrics"] = {}
    solution["metrics"]["obj1"] = obj_val
    return solution


params = {}
params['x1'] = {'type': 'categorical', 'space': 'decision', 'domain': ["A", "B", "C"], 'init': 'B'}
params['x2'] = {'type': 'float', 'space': 'decision', 'min': 0.0, 'max': 1.0, 'init': 0.5}
params['x3'] = {'type': 'float', 'space': 'decision', 'min': 0.0, 'max': 1.0, 'init': 0.5}
params['x4'] = {'type': 'integer', 'space': 'decision', 'min': 1, 'max': 10, 'step': 1, 'init': 5}

problem = {
    'problem_name': 'Mixed search space',
    'parameters': params,
    'metrics': {'obj1': {'type': 'objective', 'goal': 'minimize'}},
}

worker = indiesolver.indiesolver()
worker.initialize("indiesolver.com", 8080, "dg8f5a0dd9ed")
reply = worker.create_problem(problem)
if reply["status"] != "success":
    print(reply)
    exit()
# problem = worker.ask_problem_description('Mixed search space')
# worker.set_problem(problem)


for iteration in range(0, 100):
    reply = worker.ask_new_solutions(1)
    solutions = {}
    solutions["solutions"] = []
    if reply["status"] == "success":
        for solution in reply["solutions"]:
            solutions["solutions"].append(evaluate(solution))
        worker.tell_metrics(solutions)
        print(reply)
    else:
        print(reply)
        exit()
