# Roadmap

This page says what the `pySDC` developers intend to work on in the next year, and what they do not intend to do. It
is reviewed at least once a year and whenever plans change. Last review: November 2026.

`pySDC` is research software: much of what happens in it comes from the research projects of its developers and is
not planned a year ahead. The points below are the ones that affect users of the library.

## Planned

- **Supported Python and NumPy versions.** `pySDC` follows the Python versions its main dependencies support: the CI
  moves to Python 3.11 to 3.15 and drops 3.10. NumPy 1 and NumPy 2 both stay supported as long as optional backends
  still depend on NumPy 1.
- **Signed releases.** Releases are published to PyPI by GitHub Actions with signed attestations, and every release
  comes with software bills of materials (SBOMs); see the [security policy](./SECURITY.md).
- **Finite elements through Firedrake.** Firedrake is the finite element backend after the removal of FEniCS in
  version 5.10. Problem classes, transfer operators and tests for it are extended as projects need them.
- **Mixed and reduced precision.** Continue to investigate where SDC can use lower floating-point precision without
  losing accuracy, and where that pays off on GPUs ([#511](https://github.com/Parallel-in-Time/pySDC/issues/511)).
- **Version 6.** A more generic interface for problem classes
  ([#254](https://github.com/Parallel-in-Time/pySDC/issues/254)) and the rework of the problem classes to follow it
  ([#255](https://github.com/Parallel-in-Time/pySDC/issues/255)). Because this breaks the API, it is developed on the
  `v6` branch, as described in the [pull request guide](./docs/contrib/01_pull_requests.md).
- **New projects.** Research that uses `pySDC` is added as projects in `pySDC/projects`, with tests and a page on the
  website.

## Not planned

- **A port to FEniCSx (dolfinx).** Firedrake covers the finite element use cases.
- **Compiled extensions.** `pySDC` stays a pure Python package. Speed comes from the libraries the problem classes
  use (NumPy, SciPy, CuPy, PETSc, Firedrake), not from compiled code in `pySDC` itself.
- **Support for Python versions that are no longer maintained upstream.**
- **Multistep methods for comparison** ([#327](https://github.com/Parallel-in-Time/pySDC/issues/327)) are welcome as a
  contribution, but nobody plans to work on them.

## Changing this roadmap

Proposals go through a GitHub issue or a pull request to this file. The maintainers decide, as described in the
[governance](./GOVERNANCE.md).
