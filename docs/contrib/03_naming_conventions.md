# Naming conventions in pySDC

> :scroll: Those rules may not be enforced by the current implementation of pySDC. However, they should be enforced for any contribution.

Naming convention are mostly inspired from the [PEP-8 guidelines](https://peps.python.org/pep-0008/), even if some of them may be different. Of course, strictly following those rules is not always the best solution, as Guido Von Rossum's key insight states :

> _A Foolish Consistency is the Hobgoblin of Little Minds_

The most important idea at the end is to find a optimal compromise between

- readability : _Can someone else easily read and understand my code ?_
- effectiveness : _Does my code avoid kilometers-long lines to do simple things ?_

Both aspects are interdependent to ease maintaining/development of any code and improve its attractiveness to potential users.

## First definitions

Possible Naming formats :

- all-lowercase : `variablenamelikethis`
- snake_case : `variable_name_like_this`
- PascalCase : `VariableNameLikeThis`
- camelCase : `variableNameLikeThis`
- all-uppercase with underscore : `VARIABLE_NAME_LIKE_THIS`
- all-uppercase with minus : `VARIABLE-NAME-LIKE-THIS`

## Packages and modules names

Modules should have short, all-lowercase names. Underscores can be used in the module name if it improves readability (_i.e_ use snake_case only if it helps,
else try to stick to all-lowercase).
Python packages should also have short, all-lowercase names, although the use of underscores is discouraged.

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

Function (or method) and variable names should use camelCase formatting, and same goes for function arguments. For instance :

```python
tLeft = 1
quadType = 'LEGENDRE'

def computeFejerRule(nNodes):
    # ...

class NodeGenerator():
    def getOrthogPolyCoeffs(self, nCoeffs):
        # ...
```

:scroll: A few additional notes :

1. In general, shorter name (eventually with abbreviations) should be favored, **as long as it does not deteriorate understandability**. For instance `getOrthogPolyCoeffs` rather than `getOrthogonalPolynomialCoefficients`.
2. Suffix `s` for plural should be used even with abbreviations for consistency (_e.g_ `nCoeffs`, `nNodes`, ...).
3. Acronyms can be used to simplify variable names, but **try not to start with it**. For instance, favor `jacobiMSSDC` or `multiStepSDCJacobi` rather than `MSSDCJacobi`. In general, acronyms should be put at the end of variable names.
4. Underscore can exceptionally be used at the end of variable names when it make readability better and ease further developments. In that case, the characters after the underscore **should be all-uppercase with underscore** (minus is not allowed by Python syntax). For instance when defining the same method with different specializations :

```python
class MySweeper(Sweeper):

    def __init__(self, initSweep):
        try:
            self.initSweep = getattr(self, f'_initSweep_{initSweep}')
        except AttributeError:
            raise NotImplementedError(f'initSweep={initSweep}')

    def _initSweep_COPY(self):
        pass

    def _initSweep_SPREAD(self):
        pass

    # ... other implementations for initSweep
```

## Private and public attributes

There is no such thing as private or public attributes in Python. But some attributes, if uses only within the object methods, can be indicated as private using the `_` prefix. For instance :

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

:scroll: In general, variable name starting with double underscore `__` are usually left for Python built-in names, _e.g_ `__dict__`, `__init__`, ...

## Constants

Constants are usually defined on a module level and written in all-uppercase with underscores (all-uppercase with minus are not allowed by Python syntax). Examples :

```python
NODE_TYPES = ['EQUID', 'LEGENDRE', 'CHEBY-1', 'CHEBY-2', 'CHEBY-3', 'CHEBY-4']
QUAD_TYPES = ['GAUSS', 'RADAU-LEFT', 'RADAU-RIGHT', 'LOBATTO']
```

For _constant string values_, however, favor the use of all uppercase with minus, _e.g_ `RADAU-RIGHT`, `LEGENDRE-NUMPY` to distinguish those from constants names.

:bell: When constants are used, for instance, to select method specializations (with suffix using all-uppercase with underscore), it is probably better to keep all-uppercase with minus for constant string values and add a character replacement in between, for instance :

```python
class MySweeper(Sweeper):

    def __init__(self, initSweep):
        try:
            self.initSweep = getattr(self, f'_initSweep_{initSweep.replace('-','_')}')
        except AttributeError:
            raise NotImplementedError(f'initSweep={initSweep}')

    def _initSweep_COPY_PASTE(self):
        pass

    def _initSweep_SPREAD_OUT(self):
        pass

    # ... other implementations for initSweep
```

:arrow_left: [Back to Continuous Integration](./02_continuous_integration.md) ---
:arrow_right: [Next to Custom Implementations](./04_custom_implementations.md)