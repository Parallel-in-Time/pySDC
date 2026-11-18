# Governance

`pySDC` is developed by researchers from several institutions who use it for their own work on spectral deferred
corrections (SDC), parallel-in-time integration and related methods. This document describes the roles in the
project, who holds them, and how decisions are made.

## Roles

### Contributors

Anyone who opens an issue, comments on one, or proposes a change through a pull request. Contributors are asked to

- follow the [contribution guidelines](./CONTRIBUTING.md) and the [code of conduct](./CODE_OF_CONDUCT.md),
- add tests for the functionality they add or the bugs they fix,
- keep their pull request passing the continuous integration (CI) tests and respond to review comments.

Each project in `pySDC/projects` belongs to the contributors who wrote it. They know it best and are asked to review
changes to it.

### Core developers

Contributors with write access to the repository. In addition to what contributors do, core developers

- review pull requests and merge them,
- triage issues: label them, answer questions, close what is resolved or out of scope,
- keep the CI working when a new version of a dependency or of Python breaks it.

Current core developers: [@tlunet](https://github.com/tlunet), [@ruthschoebel](https://github.com/ruthschoebel),
[@jakob-fritz](https://github.com/jakob-fritz), [@brownbaerchen](https://github.com/brownbaerchen).

### Maintainers

Core developers with administrator rights on the repository. In addition to what core developers do, maintainers

- publish releases, following the [release guide](./docs/contrib/07_release_guide.md),
- handle security reports as described in the [security policy](./SECURITY.md),
- manage repository settings, access rights and the services connected to the repository (PyPI publishing, the
  website, Codecov),
- decide when no consensus can be reached (see below).

Current maintainers: [@pancetta](https://github.com/pancetta) (Robert Speck, Forschungszentrum Jülich) and
[@danielru](https://github.com/danielru) (Daniel Ruprecht, Hamburg University of Technology).

## How decisions are made

- **Day-to-day changes** go through pull requests. A core developer or maintainer merges a pull request once the CI
  passes and the review comments are resolved. Changes to `pySDC/core` and changes that alter the user interface need
  the approval of at least one core developer or maintainer other than the author.
- **Larger changes**, such as API-breaking changes, dropping support for a Python or NumPy version, or adding a
  required dependency, are first discussed in a GitHub issue. API-breaking changes are developed on a release branch
  (`v6`, ...) whose roadmap is written down in a dedicated issue, as described in the
  [pull request guide](./docs/contrib/01_pull_requests.md).
- The project works by consensus among the people involved in a discussion. If no consensus is reached, the
  maintainers decide.

## Becoming a core developer or maintainer

Contributors who have contributed regularly over some time and reviewed others' work are invited by the maintainers
to become core developers. New maintainers are appointed by the current maintainers from among the core developers.
Core developers and maintainers may step down at any time.

## Continuity

The two maintainers work at different institutions and both hold administrator rights on the repository. Releases
are published to PyPI by a GitHub Actions workflow through trusted publishing, so neither maintainer's personal
credentials are needed to make a release. If one maintainer becomes unavailable, the other can continue to merge
pull requests, close issues, publish releases and appoint a new maintainer.

## Changing this document

Changes to this document are proposed through a pull request and need the approval of a maintainer.
