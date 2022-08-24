import numpy as np
from scipy.special import factorial


def get_linear_multistep_method(steps, u_signature, f_signature, checks=False):
    '''
    Derive a general linear multistep method from step sizes and a signature. This function will provide a consistent
    linear multistep method by cancelling terms in Taylor expansions, but note that this must not be stable!

    The resulting coefficients must be multiplied with the corresponding value of u or f and then all must be summed to
    get a numerical solution to the initial value problem.

    Since we can cancel as many terms in the Taylor expansion as we have entries in the signature, that will also be
    the order of consistency of our method.

    We check if our method is consistent and zero stable, which together means it is convergent.
    However, some of the methods that we generate are not A stable. As it turns out, according to Dahlquist's second
    barrier theorem, there are no A-stable LMMs of order greater than 2.

    Args:
        steps (list): The step sizes between the multiple steps that are used
        u_signature (list): A list containing which solutions at which steps should be used. Set to 1, for all times at
                            which you want to use the solution and to 0 at all other times
        f_signature (list): Analogue to u_signature for the right hand side evaluations
        checks (bool): Perform some checks on stability and convergence

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

    if checks:
        # check that the method is consistent
        check_linear_difference_operator(u_coeff, f_coeff, steps)

        # check if our method is zero stable
        verify_root_condition(first_characteristic_polynomial(u_coeff))

        # Check if the method is stable for h*lambda=-1
        p = stability_polynomial(u_coeff, f_coeff, -1.)
        strict_root_condition(p)

    return u_coeff, f_coeff


def first_characteristic_polynomial(u_coeff, r=1):
    '''
    The first characteristic polynomial of a linear multistep method is equal to the coefficients of umultiplied with
    powers of r.

    Args:
        u_coeff: The alpha coefficients for u of the LMM in order of descending time difference to the solution we want
        r: The variable of the polynomial

    Returns:
        numpy.ndarray: List containing the polynomial in r. Set r=1 to get the coefficients.
    '''
    j = np.arange(len(u_coeff))
    rho = np.zeros_like(u_coeff)
    rho = -u_coeff * r**j
    rho[-1] = r**len(u_coeff)
    return rho[::-1]  # reverse the order to go along with usual definitions


def second_characteristic_polynomial(f_coeff, r=1):
    '''
    The second characteristic polynomial of a linear multistep method is equal to the coefficients multiplied with
    powers of r.

    Args:
        f_coeff: The alpha coefficients for f of the LMM in order of descending time difference to the solution we want
        r: The variable of the polynomial

    Returns:
        numpy.ndarray: List containing the polynomial in r. Set r=1 to get the coefficients.
    '''
    j = np.arange(len(f_coeff))
    sigma = np.zeros_like(f_coeff)
    sigma = f_coeff * r**j
    return sigma[::-1]  # reverse the order to go along with usual definitions


def verify_root_condition(rho):
    '''
    For a linear multistep method to be convergent, we require that all roots of the first characteristic polynomial
    are distinct and have modulus smaller or equal to one.
    This root condition implies that the method is zero stable and Dahlquist's theorem states that a zero stable and
    consistent method is convergent. If we can also show that the method is consistent, we have thus shown it is
    convergent.

    Args:
        rho (numpy.ndarray): Coefficients of the first characteristic polynomial

    Returns:
        bool: Whether the root condition is satisfied.
    '''
    # compute the roots of the polynomial
    roots = np.roots(rho)

    # check the conditions
    roots_distinct = len(np.unique(roots)) == len(roots)
    # give some leeway because we introduce some numerical error when computing the roots
    modulus_condition = all(abs(roots) <= 1. + 10. * np.finfo(float).eps)

    # raise errors if we violate one of the conditions
    assert roots_distinct, "Not all roots of the first characteristic polynomial of the LMM are distinct!"
    assert modulus_condition, "Some of the roots of the first characteristic polynomial of the LMM have modulus larger \
one!"
    return roots_distinct and modulus_condition


def check_linear_difference_operator(u_coeff, f_coeff, steps):
    '''
    Check if the linear multistep method is consistent by doing a Taylor expansion and testing if all terms cancel
    except for the first, which should be one.

    Args:
        u_coeff (numpy.ndarray): Coefficients for u in the LMM
        f_coeff (numpy.ndarray): Coefficients for f in the LMM
        steps (numpy.ndarray): Steps from point of expansion

    Returns:
        None
    '''
    order = len(steps)
    taylor_coeffs = np.zeros((len(u_coeff) + len(f_coeff), order))

    # fill in the coefficients
    for i in range(order):
        # get expansions of u
        if u_coeff[i] != 0:
            taylor_coeffs[i, :] = u_coeff[i] * taylor_expansion(steps[i], order)

        # get expansions of f
        if f_coeff[i] != 0:
            taylor_coeffs[order + i, 1:] = f_coeff[i] * taylor_expansion(steps[i], order - 1)

    # check that all is well
    L = np.sum(taylor_coeffs, axis=0)
    want = np.zeros_like(L)
    want[0] = 1.
    assert all(np.isclose(L, want)), "Some derivatives do not cancel in the Taylor expansion!"

    return None


def taylor_expansion(step, order):
    '''
    Get coefficients of a Taylor expansion.

    Args:
        step (float): Time difference from point around which we expand
        order (int): The order up to which we want to expand

    Returns:
        numpy.ndarray: List containing the coefficients of the derivatives of u in the Taylor expansion
    '''
    j = np.arange(order)
    return step**j / factorial(j)


def stability_polynomial(u_coeff, f_coeff, h_hat):
    '''
    Construct the stability polynomial for a value of h_hat = h * lambda.

    Args:
        u_coeff (numpy.ndarray): Coefficients for u in the LMM
        f_coeff (numpy.ndarray): Coefficients for f in the LMM
        h_hat (float) Parameter where you want to check stability

    Returns:
        numpy.ndarray: List containing the coefficients of the stability polynomial
    '''
    rho = first_characteristic_polynomial(u_coeff)
    sigma = second_characteristic_polynomial(f_coeff)
    return rho - h_hat * sigma


def strict_root_condition(p):
    '''
    Check whether the roots of the polynomial ae strictly smaller than one.

    Args:
        p (numpy.ndarray): Coefficients for the polynomial

    Returns:
        None
    '''
    roots = np.roots(p)
    assert all(abs(roots) < 1), "Polynomial does not satisfy strict root condition!"
    return None
