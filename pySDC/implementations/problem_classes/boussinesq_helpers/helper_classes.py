class logging(object):
    def __init__(self):
        self.solver_calls = 0
        self.iterations = 0
        self.nsmall = 0

    def add(self, iterations):
        self.solver_calls += 1
        self.iterations += iterations


class Callback(object):
    def getresidual(self):
        return self.residual

    def getcounter(self):
        return self.counter

    def __init__(self):
        self.counter = 0
        self.residual = 0.0

    def __call__(self, residuals):
        self.counter += 1
        self.residual = residuals
