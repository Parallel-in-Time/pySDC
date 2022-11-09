# Preconditioners
The major goal of this project is to derive preconditioners using optimization and adaptivity.

We made some considerations of how to evaluate the contraction factors of preconditioners using Fouriour transform [here](data/notes/Fourier.md).

We discussed details of the optimization [here](data/notes/optimization.md).

## Things we learned so far
 - Preconditioners are better at handling stiff problems when we initialized the intermediate solutions during optimization randomly.

## TODOs
 - Make more elaborate objective functions
 - Can we use the extrapolation error estimate in objective functions?
 - Use Dahlquist problem for optimization