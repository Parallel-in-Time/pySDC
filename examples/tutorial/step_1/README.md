Step-1: A first spatial problem
===============================

In this step, we create a spatial problem and play with it a little bit. No SDC, no multi-level here so far.

Part A
------
We start as simple as possible by creating a first, very simple problem using one of the `problem_classes`, namely `HeatEquation_1D_FD`. 
This basically consists only of the matrix `A`, which represents a finite difference discretization of the 1D Laplacian.
This is tested for one particular example.

Important things to note:

* Many (most) parameters for pySDC are passed using dictionaries
* Data types are encapsulated: the real values are stored in `values`, meta-information can be stored separately in the data structure
* We happily pass classes around so that they can be used to instantiate things within subroutines


Part B
------
We now do a more thorough test of the accuracy of the spatial discretization.
We loop over a set of `nvars`, compute the Laplacian of our test vector and look at the error.
Then, the order of accuracy in space is checked by looking at the numbers (part `B1`) and by looking at "points on a line" (part `B2`).

Important things to note:

* You better test your operators.. use nvars > 2**16 and things will break!
* Add your results into a dictionary for later usage. Use IDs to find the data! Also, use headers to store meta-information.


Part C
------
Here, we set up our first collocation problem using one of the `collocation_classes`, namely `GaussRadau_Right`.
Using the spatial operator, we can build the **collocation problem** which in this case is linear.
This fully coupled system is then solved directly and the error is compared to the exact solution.
 
Important things to note:

* The collocation matrix `Q` is and will be always relative to the temporal domain [0,1]. Use `dt` to scale it appropriately. 
* Although convenient to analyze, the matrix formulation is not suited for larger (in space or time) computations. 
All remaining computations in pySDC are thus based on decoupling space and time operators (i.e. no Kronecker product).
* We can use the `u_exact` routine here to return the values at any given point in time. 
It is recommended to include either an initialization routine or the exact solution (if applicable) into the problem class.
* This is where the fun with parameters comes in: How many DOFs do we need in space? How large/small does dt have to be? What frequency in space is fine? ...


Part D
------
As for the spatial operator, we now test the accuracy in time.
This time, we loop over a set of `dt`, compute the solution to the collocation problem and look at the error.

Important things to note:

* We take a large number of DOFs in space, since we need to beat 5th order in time with a 2nd order stencil in space.
* Orders of convergence are not as stable as for the space-only test. 
One of the problems of this example is that we are actually trying to compute 0 very, very thorougly... 



