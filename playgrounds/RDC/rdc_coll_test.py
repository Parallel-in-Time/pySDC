import numpy as np

from playgrounds.RDC.equidistant_RDC import Equidistant_RDC


def main():
    """
    A simple test program print the collocation matrix for RDC
    """

    # instantiate collocation class, relative to the time interval [0,1]
    coll = Equidistant_RDC(num_nodes=16, tleft=0, tright=np.pi)

    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True)
    print(coll.Qmat)

if __name__ == "__main__":
    main()