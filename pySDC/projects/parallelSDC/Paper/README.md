# Numerical experiment scripts for the parallel SDC paper

_Selected problems are : Van der Pol oscilator, Lorenz system_

Common utility scripts :

- [`utils.py`](./utils.py) : utility functions to get numerical solutions (with SDC, exact or collocation)

> :mega: Each script can generate pdf plots with name starting with the name of the script.

## Van der Pol

- [`vanderpol_period.py`](./vanderpol_period.py) : determine period for different `mu` values, and plot scaled exact solution on one period
- [`vanderpol_accuracy.py`](./vanderpol_accuracy.py) : investigate accuracy of diagonal SDC variants on the different Van der Pol problems
