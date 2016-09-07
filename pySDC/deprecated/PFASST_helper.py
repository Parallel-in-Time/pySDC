class switch(object):
    """
    Helper class for using case/switch statements in Python (not necessary, but easier to read)
    """
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration

    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False


def generate_steps(num_procs,sparams,description):
    """
    Routine to easily generate a block of steps to run in (pseudo-)parallel

    This function is the quick way to get a block of steps. Each step is treated the same way here, i.e. using the same
    hierarchy and the same parameter set per step.

    Args:
        num_procs: number of (virtual) processors
        sparams: parameters for the steps
        description: description dictionary for the hierarchy

    Returns:
        block of steps (list)
    """

    from pySDC import Step as stepclass

    MS = []
    # simply append step after step and generate the hierarchies
    for p in range(num_procs):
        MS.append(stepclass.step(sparams))
        MS[-1].generate_hierarchy(description)

    return MS


def check_convergence(S):
    """
    Routine to determine whether to stop iterating (currently testing the residual and the max. number of iterations)

    Args:
        S: current step

    Returns:
        converged, true or false

    """

    # do all this on the finest level
    L = S.levels[0]

    # get residual and check against prescribed tolerance (plus check number of iterations
    res = L.status.residual
    converged = S.status.iter >= S.params.maxiter or res <= L.params.restol

    return converged