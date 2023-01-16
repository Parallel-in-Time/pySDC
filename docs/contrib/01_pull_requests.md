# Recommendations for pull requests

Contributions on the `pySDC` code is expected to be done through pull requests from personal (public) forked repositories. A few core developers can eventually push maintenance commits directly to the main repository. However (even for core developers), it is highly recommended to add specific contribution trough dedicated merge requests from forks.

## Contributing to the main branch

The main version of `pySDC` is hosted in the `master` branch, on which any contributor can propose pull requests. Those can consist on :

- bug fixes and code corrections (_e.g_ solving one of the current [issues](https://github.com/Parallel-in-Time/pySDC/issues))
- addition or improvement of documentation
- improvement of existing functionalities (performance, accuracy, usage, ...)
- addition of new functionalities and applications
- improvement of CI test routines

Pull request should comes from forks branches with a name specific to the contribution. For instance :

```
# branch name :
issue214  # to solve issue 214
awesome_new_project  # to add a new project
some_feature  # to add a new feature (implementation, ...)
```

> :scroll: Favor the use of _short name_ for branch, using _lower case_ and eventually underscores to ease readability.

Those changes should be compatible with the existing API (_i.e_ not break it), and **avoid any change** in the current user interface. In particular, it should not modify default values for parameters or remove attributes of existing classes. But new attributes or parameters can be added with pre-set default values, and new classes can be added in the `pySDC.implementations` module.


## Release development branches

Branches with name starting with `v[...]` are development branches for the next releases of `pySDC` (_e.g_ `v5`, `v6`, ...). Those may introduce API-breaking changes (user interface, structure of core classes) that would force re-writing application scripts using `pySDC` (_e.g_ tutorials, projects, ...). Contribution to those branches are done by core developers, but anyone can also propose pull requests on those branches once the roadmap and milestones for the associated release has been written down in a dedicated issue.
Such branches are merged to `master` when ready.

> :scroll: Pull request to those branches can be done from fork branches using the **same name** as the release branch.

## Feature development branches

Additional branches starting with the prefix `dev/[...]` can be used to add new features, that cannot be added with only one pull request (for instance, when several developers work on it).
Those could eventually be merged into master if they don't break the API, or to the next release branch if they do.

> :scroll: Pull request to those branches can be done from fork branches using the **same name** as the feature branch.

[:arrow_left: Back to Contributing Summary](./../../CONTRIBUTING.md) ---
[:arrow_right: Next to Continuous Integration](./02_continuous_integration.md)