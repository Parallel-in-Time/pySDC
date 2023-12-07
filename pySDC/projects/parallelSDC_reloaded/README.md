# Numerical experiment scripts for the parallel SDC paper

_Selected problems are : Van der Pol oscilator, Lorenz system_

Common utility scripts :

- [`utils.py`](./utils.py) : utility functions to get numerical solutions with SDC, or reference solutions with scipy (exact)

> :mega: Each script can generate pdf plots with name starting with the name of the script, stored in the [`fig`](./fig/) folder

## Van der Pol

- [`vanderpol_setup.py`](./vanderpol_setup.py) : determine period for different `mu` values, and plot scaled exact solution on one period
- [`vanderpol_accuracy.py`](./vanderpol_accuracy.py) : investigate accuracy of diagonal SDC variants on the different Van der Pol problems

## Lorenz

- [`lorenz_setup.py`](./lorenz_setup.py) : plot numerical solution of Lorenz, and determine numerically the final time of a given number of revolution periods
- [`lorenz_accuracy.py`](./lorenz_accuracy.py) : investigate accuracy of diagonal SDC variants on the Lorenz system on a given simulation time (given number of revolution periods)

## Prothero-Robinson

- [`protheroRobinson_setup.py`](./protheroRobinson_setup.py) : numerical solution for linear and non-linear case
- [`protheroRobinson_accuracy.py`](./protheroRobinson_accuracy.py) : investigate accuracy of diagonal SDC variants
