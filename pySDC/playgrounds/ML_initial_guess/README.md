Machine learning initial guesses for SDC
----------------------------------------

Most linear solves in SDC are performed in the first iteration. Afterwards, SDC providing good initial guesses is actually one of its strengths.
To get a better initial guess for "free", and to stay hip, we want to do this with machine learning.

This playground is very much work in progress!
The first thing we did was to build a simple datatype for PyTorch that we can use in pySDC. Keep in mind that it is very inefficient and I don't think it works with MPI yet. But it's good enough for counting iterations. Once we have a proof of concept, we should refine this.
Then, we setup a simple heat equation with this datatype in `heat.py`.
The crucial new function is `ML_predict`, which loads an already trained model and evaluates it.
This, in turn, is called during `predict` in the sweeper. (See `sweeper.py`)
But we need to train the model, of course. This is done in `ml_heat.py`.

How to move on with this project:
=================================
The first thing you might want to do is to fix the neural network that solves the heat equation. Our first try was too simplistic.
The next thing would be to not predict the solution at a single node, but at all collocation nodes simultaneously. Maybe, actually start with this.
If you get a proof of concept, you can clean up the datatype, such that it is even fast.
You can do a "physics-informed" learning process of predicting the entire collocation solution by means of the residual. This is very generic, actually.
