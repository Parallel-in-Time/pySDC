# Publishing a new release

## Base conventions

For each version update (a.k.a **releases**), we use the following denomination :

- **patch** (from `*.*.{i}` to `*.*.{i+1}`) : minor modifications, bugfixes, code reformating, small new features, new projects or playgrounds, new tests
- **minor** (from `*.{i}.*` to `*.{i+1}.0`) : addition of new major features, minor code structure changes without too much impact on the API (backward compatible)
- **major** (from `{i}.*.*` to `{i+1}.0.0`) : major changes in code structure, design, and API, with changes potentially breaking backward compatibility 

## Release Pipeline

First, create a `new-release` branch, either on your fork or on the main `pySDC` repo. Then, in on commit :

1. modify the project version number in `pyproject.toml`
2. modify the documentation release number in `docs/source/conf.py`, and the version number for minor and major release
3. modify the version number and release date in `CITATION.cff`
4. (for minor and major release **only**) add the release description in the `CHANGELOG.md` file

Commit with the message : `bump version to x.x.x` where `x.x.x` is the new version. 
Then create a pull request, and once all tests passed, you can `Merge and Squash`,
possibly adding your initials as prefix of the final commit message.

Finally, [draft a new release](https://github.com/Parallel-in-Time/pySDC/releases/new) associated to a new tag 
`v*.*.*` (with `*.*.*` the new version, and the ` + Create new tag: ... on publish` button).
Add a comprehensive summary of the main changes, with appropriate thanks to all the contributors (cf previous releases), and publish it. This will trigger automatically a release update on [Zenodo](https://zenodo.org/doi/10.5281/zenodo.594191) and [PyPI](https://pypi.org/project/pySDC/).

:arrow_left: [Back to adding Project](./06_new_project.md) ---
:arrow_up: [Contributing Summary](./../../CONTRIBUTING.md) ---
:arrow_right: [Next to a cute picture of cat](https://www.vecteezy.com/photo/2098203-silver-tabby-cat-sitting-on-green-background)