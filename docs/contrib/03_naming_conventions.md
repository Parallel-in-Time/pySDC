# Naming conventions in pySDC

> :scroll: Those rules may not be enforced by the current implementation of pySDC. However, they should be enforced for any contribution.

Naming convention are mostly inspired from the [PEP-8 guidelines](https://peps.python.org/pep-0008/), even if some of them may be different. Of course, strictly following those rules is not always the best solution, as Guido Von Rossum's key insight states :

> _A Foolish Consistency is the Hobgoblin of Little Minds_

The most important idea at the end is to find a optimal compromise between

- readability : _Can someone else easily read and understand my code ?_
- effectiveness : _Does my code avoid kilometers-long lines to do simple things ?_

Both aspects are interdependent to ease maintaining/development of any code and improve its attractiveness to potential users.

## Packages and modules names

Modules should have short, all-lowercase names. Underscores can be used in the module name if it improves readability. Python packages should also have short, all-lowercase names, although the use of underscores is discouraged.

## Class names

Class names should use PascalCase formatting, for instance :

```python
class AdvectionDiffusion(Problem):
    pass
```

The shorter, the better. Also, exception class names should end with the suffix `Error`, for instance

```python
class ParameterError(Exception):
    pass
```

## Function and variables names

Function (or method) and variable names should be lowercase, with words separated by underscores as necessary to improve readability.
Same goes for function arguments. For instance :

```python
tleft = 1
quad_type = 'LEGENDRE'

def compute_fejer_rule(num_nodes):
    # ...

class NodeGenerator():
    def get_orthog_poly_coefficients(self, num_coeff):
        # ...
```

> :bell: In general, shorter name should be favored, **as long as it does not deteriorate understandability**. For instance, using `get_orthog_poly_coeffs` rather than `get_orthogonal_polynomial_coefficients`. Acronyms can also be used, but with caution (for instance, `mssdc_jac` may not be very explicit for `multistep_sdc_jacobian`).

## Private and public attributes

There is no such thing as private or public attributes in Python. But some attributes, if uses only within the object method, can be indicated as private using the `_` prefix. For instance :

```python
class ChuckNorris():

    def __init__(self, param):
        self.param = param

    def _think(self):
        print('...')

    def act(self):
        if self.param == 'doubt':
            self._think()
        print('*?%&$?*§"$*$*§#{*')
```

## Constants

Constants are usually defined on a module level and written in all capital letters with underscores separating words.
Examples :

```python
NODE_TYPES = ['EQUID', 'LEGENDRE', 'CHEBY-1', 'CHEBY-2', 'CHEBY-3', 'CHEBY-4']
QUAD_TYPES = ['GAUSS', 'RADAU-LEFT', 'RADAU-RIGHT', 'LOBATTO']
```

[:arrow_left: Back to Continuous Integration](./02_continuous_integration.md) ---
[:arrow_right: Next to Custom Implementations](./04_custom_implementations.md)