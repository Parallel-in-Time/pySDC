
import indiesolver
import json
import numpy as np

params = {}

problem = {'problem_name': 'bla',
           'parameters': params,
           'metrics': { 'rho' : { 'type': 'objective', 'goal': 'minimize'}}}

worker = indiesolver.indiesolver()
worker.initialize("indiesolver.com", 8080, "dg8f5a0dd9ed")
reply = worker.create_problem(problem)

problem_names = worker.ask_problems()
for problem_name in problem_names['problems']:
    problem = worker.ask_problem_description(problem_name)
    problem['problem_name'] = problem_name
    worker.set_problem(problem)
    pending_solutions = worker.ask_pending_solutions()
    print(pending_solutions)
    for metric in problem['metrics']:
        for solution in pending_solutions['solutions']:
            solution['metrics'] = {}
            solution['metrics'][metric] = 1e+10
        worker.tell_metrics(pending_solutions)

