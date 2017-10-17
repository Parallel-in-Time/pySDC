from pySDC.tutorial.step_6.A_classic_vs_multigrid_controller import main as main_A


def main():
    """
    A simple test program to do check PFASST for odd numbers of processes
    """
    main_A(num_proc_list=[3, 5, 7, 9], fname='step_6_B_out.txt')


if __name__ == "__main__":
    main()
