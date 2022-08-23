import numpy as np
from scipy.special import factorial


def get_linear_multistep_method(steps, u_signature, f_signature):
    '''
    Derive a general linear multistep method from step sizes and a signature. This function will provide a consistent
    linear multistep method by cancelling terms in Taylor expansions, but note that this must not be convergent!

    The resulting coefficients must be multiplied with the corresponding value of u or f and then all must be summed to
    get a numerical solution to the initial value problem.

    Since we can cancel as many terms in the Taylor expansion as we have entries in the signature, that will also be
    the order of consistency of our method.

    Args:
        steps (list): The step sizes between the multiple steps that are used
        u_signature (list): A list containing which solutions at which steps should be used. Set to 1, for all times at
                            which you want to use the solution and to 0 at all other times
        f_signature (list): Analogue to u_signature for the right hand side evaluations

    Returns:
        list: Coefficients for u
        list: Coefficients for f
    '''
    n_u = np.sum(u_signature, dtype=int)
    n_f = np.sum(f_signature, dtype=int)
    n = n_u + n_f  # number of available values
    j = np.arange(n)  # index variable
    inv_fac = 1. / factorial(j)  # compute inverse factorials once to use in Taylor expansions

    # build a matrix containing the Taylor coefficients
    A = np.zeros((n, n))

    # fill the entries for u
    for i in range(n_u):
        A[:, i] = steps[u_signature > 0][i]**j * inv_fac

    # fill the entries for f
    for i in range(n_f):
        A[1:, i + n_u] = steps[f_signature > 0][i]**j[:-1] * inv_fac[:-1]

    # build a right hand side vector for solving the system
    b = np.zeros(n)
    b[0] = 1.

    # solve the linear system
    coeff = np.linalg.solve(A, b)

    # distribute the coefficients
    u_coeff = np.zeros_like(u_signature)
    u_coeff[u_signature > 0] = coeff[0: n_u]
    f_coeff = np.zeros_like(f_signature)
    f_coeff[f_signature > 0] = coeff[n_u:]

    return u_coeff, f_coeff
