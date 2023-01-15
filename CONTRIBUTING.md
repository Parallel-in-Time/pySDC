# How to contribute to pySDC

1. [Pull Requestion recommendations](./docs/contrib/01_pull_requests.md)
2. [Continuous Integration](./docs/contrib/02_continuous_integration.md)
3. [Naming Conventions](./docs/contrib/03_naming_conventions.md)
4. [Custom Implementations](./docs/contrib/04_custom_implementations.md)

Developments on the `pySDC` code use the classical approach of forks and pull requests from Github.
There is an [extended documentation](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/about-collaborative-development-models) on this aspect (that you can skip if you are already used to it). In addition, some recommendations for pull requests are given [here](./docs/contrib/01_pull_requests.md).

Additionnaly, a _few_ rules are set to enforce code readability, consistency and reliability. Some of them are automatically tested with each commit, and summarized in the page on [continuous integration (CI)](./docs/contrib/02_continuous_integration.md).
Others are specific conventions chosen for the pySDC library, that may follow Python standards (or not ...), detailed in the [naming conventions page](./docs/contrib/03_naming_conventions.md).

Finally, while `pySDC` provides many base functionalities that implement classical flavors of SDC, it also allows problem-specific applications through Object-Oriented Programming (OOP) and the implementation of custom inherited classes.
This follows a specific OOP framework, for which more details are given [here](.(docs/contrib/../../docs/contrib/04_custom_implementations.md)).