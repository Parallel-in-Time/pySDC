# Continuous Integration in pySDC

Any commit in `pySDC` are tested within by GitHub continuous integration (CI). You can see in in the [action panel](https://github.com/Parallel-in-Time/pySDC/actions) the tests for each branches.
Those tests can be divided in two main categories : [code linting](#code-linting) and [code testing](#code-testing).
Finally, the CI also build artifacts that are used to generate the documentation website (see http://parallel-in-time.org/pySDC/), more details given in the [documentation generation](#documentation-generation) section.

## Code linting

Code style linting is performed using [black](https://black.readthedocs.io/en/stable/) and [flakeheaven](https://flakeheaven.readthedocs.io/en/latest/) for code syntax checking. In particular, `black` is used to check compliance with (most of) [PEP-8 guidelines](https://peps.python.org/pep-0008/).

Those tests are conducted for each commit (even for forks), but you can also run it locally in the root folder of `pySDC` before pushing any commit :

```bash
# Install required packages (works also with conda/mamba)
pip install black flakeheaven flake8-comprehensions flake8-bugbear
# First : test code style linting with black
black pySDC --check --diff --color
# Second : test code syntax with flakeheaven
flakeheaven lint --benchmark pySDC
```

> :bell: To avoid any error about formatting (`black`), you can simply use this program to reformat directly your code using the command :
>
> ```bash
> black pySDC
> ```

Some style rules that are automatically enforced :

- lines should be not longer than 120 characters
- arithmetic operators (`+`, `*`, ...) should be separated with variables by one empty space

## Code testing

This is done using [pytest](https://docs.pytest.org/en/7.2.x/), and runs all the tests written in the `pySDC/tests` folder. You can run those locally in the root folder of `pySDC` using :

```bash
# Install required packages (works also with conda/mamba)
pip install pytest<7.2.0 pytest-benchmark coverage[toml]
# Run tests
pytest -v pySDC/tests
```

> :bell: Many components are tested (core, implementations, projects, tutorials, etc ...) which make the testing quite long.
> When working on a single part of the code, you can run only the corresponding part of the test by specifying the test path, for instance :
>
> ```bash
> pytest -v pySDC/tests/test_nodes.py  # only test nodes generation
> ```

## Documentation generation

Documentation is built using [sphinx](https://www.sphinx-doc.org/en/master/).
To check its generation, you can wait for all the CI tasks to download the `docs` artifacts, unzip it and open the `index.html` file there with you favorite browser. 

However, when you are working on documentation (of the project, of the code, etc ...), you can already build and check the website locally :

```bash
# Run all tests, continuing even with errors
pytest --continue-on-collection-errors -v --durations=0 pySDC/tests
# Generate rst files for sphinx
./docs/update_apidocs.sh
# Generate html documentation
sphinx-build -b html docs/source docs/build/html
```

Then you can open `docs/build/html/index.html` using you favorite browser and check how your own documentation looks like on the website.

> :bell: **Important** : running all the tests is necessary to generate graphs and images used by the website. 
> But you can still generate the website without it : just all images for the tutorials, projects and playgrounds will be missing.
> This approach can be considered for local testing of your contribution when it does not concern parts containing images (_i.e_ project or code documentation).

:arrow_left: [Back to Pull Request Recommendation](./01_pull_requests.md) ---
:arrow_up: [Contributing Summary](./../../CONTRIBUTING.md) ---
:arrow_right: [Next to Naming Conventions](./03_naming_conventions.md)