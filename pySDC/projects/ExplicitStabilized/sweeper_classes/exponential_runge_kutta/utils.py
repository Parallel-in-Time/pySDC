import numpy as np
import numdifftools.fornberg as fornberg


if __name__ == "__main__":
    # testing the fornberg methods

    x = np.array([0.,0.5,1.])
    x0 = 0.
    
    # f(x)   = x*(1-x) = x-x**2
    # f'(x)  = 1-2*x
    # f''(x) = -2
    # f(0) = 0, f'(0) = 1, f''(0) = -2

    f = lambda x: x*(1.-x)
    fx = f(x)

    w = fornberg.fd_weights_all(x,x0,2)
    df = w @ fx 
    assert np.allclose(df,[0.,1.,-2.])
    
