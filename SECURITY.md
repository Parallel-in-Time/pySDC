# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 5.x     | :white_check_mark: |
| < 5.0   | :x:                |

We recommend using the latest version of pySDC to ensure you have the latest security updates.

## What You Can and Cannot Expect

pySDC is a library for numerical simulations. It is imported by your own Python scripts and runs with the privileges of
the user who runs them. Its security properties follow from that:

- The library (`pySDC/core`, `pySDC/implementations` and `pySDC/helpers`) opens no network connections, starts no
  other programs, evaluates no code given to it as text, and handles no passwords, tokens or keys.
- Everything pySDC computes on comes from the calling script: parameters, initial conditions, problem setups. pySDC
  treats this input as trusted, just like the script itself. It checks parameters for consistency to catch mistakes
  (for instance with a `ParameterError`), not to defend against an attacker.
- **Loading result files can run code.** `LogToFile.load` in `pySDC/implementations/hooks/log_solution.py` and many
  scripts in `pySDC/projects` read results with `pickle` or `dill`, and loading a pickle can execute arbitrary code.
  Only load files that you or someone you trust created. Files written with `pySDC.helpers.fieldsIO` contain only raw
  numbers and a fixed header, and are read without `pickle`.
- Parallel runs communicate through MPI or NCCL, as set up on your system. pySDC adds no encryption or authentication
  to this communication; all processes of a run are expected to belong to the same user.
- The code in `pySDC/projects` and `pySDC/playgrounds` is research code. Some of it writes job scripts or submits them
  to a batch system. Treat it like any other script before you run it.

## Reporting a Vulnerability

We take the security of pySDC seriously. If you believe you have found a security vulnerability, please report it to us as described below.

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to:

- **r.speck@fz-juelich.de** (Robert Speck, Project Maintainer)

You should receive a response within 5 business days. If for some reason you do not, please follow up via email to ensure we received your original message.

Please include the following information in your report (as much as you can provide):

- Type of issue (e.g., arbitrary code execution, unsafe deserialization, path traversal, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit the issue

This information will help us triage your report more quickly.

## Security Update Policy

Security updates will be released as soon as possible after a vulnerability is confirmed and a fix is available. Updates will be announced through:

- GitHub Security Advisories
- Release notes in the [CHANGELOG](./CHANGELOG.md)
- The project's GitHub Releases page

The people who report a vulnerability are credited in the advisory and the release notes, unless they ask to stay
anonymous.

## Verifying Releases

Releases from v5.10.1 on are uploaded to PyPI by the GitHub Actions workflow
[`.github/workflows/pypi.yml`](./.github/workflows/pypi.yml) through
[trusted publishing](https://docs.pypi.org/trusted-publishers/). The upload attaches a signed attestation
([PEP 740](https://peps.python.org/pep-0740/)) to every file. It is signed with [Sigstore](https://www.sigstore.dev/),
which ties the signature to this repository and the workflow run that built the file, so there is no long-lived private
key that could be stolen from PyPI or from a maintainer.

PyPI shows the attestation under "Provenance" on each file's page. To check a file yourself, use
[`pypi-attestations`](https://github.com/trailofbits/pypi-attestations), for example:

```bash
pip install pypi-attestations
pypi-attestations verify pypi --repository https://github.com/Parallel-in-Time/pySDC pypi:pysdc-<version>-py3-none-any.whl
```

The command downloads the file and its attestation from PyPI and fails unless the signature is valid and was made by a
workflow in `Parallel-in-Time/pySDC`. The wheel name in the example holds from 5.10.2 on; the wheel of 5.10.1 is named
`pysdc-5.10.1-py2.py3-none-any.whl`.

## Dependencies

pySDC uses several third-party dependencies. We monitor our dependencies for known security vulnerabilities and update them as needed. If you discover a security issue in one of our dependencies, please also report it to the respective maintainers of that dependency.

Every [release](https://github.com/Parallel-in-Time/pySDC/releases) comes with software bills of materials (SBOMs) in CycloneDX format, generated from the installed environments with the exact versions resolved on the day of the release:

- `sbom-pySDC.cdx.json`: the core install, i.e. `pip install pySDC`
- `sbom-env-<name>.cdx.json`: the optional backends in `etc/environment-<name>.yml` (PETSc, MPI, CuPy, PyTorch, ...); `sbom-env-firedrake.cdx.json` also covers Firedrake and Gusto, which are pip-installed on top of `etc/environment-firedrake.yml`
- `sbom-project-<name>.cdx.json`: the environments of the projects in `pySDC/projects/<name>`

The ones for the newest release are always at `https://github.com/Parallel-in-Time/pySDC/releases/latest/download/<file>`, e.g. [sbom-pySDC.cdx.json](https://github.com/Parallel-in-Time/pySDC/releases/latest/download/sbom-pySDC.cdx.json).

## Why These Expectations Hold

This section is the project's argument (its "assurance case") that the expectations listed above are met.

**Threat model.** pySDC runs on workstations and HPC systems as part of a user's own scripts. What an attacker could
target there is the user's account and data. Since pySDC opens no network connections, an attacker can only reach it
through what the user brings in: a result file from someone else, a modified copy of the source code, or a compromised
dependency.

**Trust boundaries.**

1. The Python process: the user's script, pySDC and its dependencies run with the same privileges and trust each other.
2. Files read from disk: they are as trustworthy as whoever wrote them, which is why loading pickles is limited to
   trusted files above.
3. Code installed from PyPI or conda-forge: pySDC itself and its dependencies.
4. The processes of a parallel run: they belong to the same user and are connected by the system's MPI or NCCL.

**Secure design principles.** pySDC needs no privileges beyond those of the user (least privilege). The library has
no network code, no calls to a shell or to other programs, and no `eval` or `exec`, so these whole classes of attacks
do not apply (economy of mechanism). Invalid or inconsistent parameters stop the run with an error instead of being
guessed (fail-safe defaults).

**Common weaknesses and how they are countered.**

- Injection and unsafe calls: the library contains no shell calls, `eval` or `exec`. [CodeQL](https://codeql.github.com/)
  checks the Python code and the GitHub Actions workflows on every push and pull request.
- Unsafe deserialization: the only place the library itself loads pickles is `LogToFile.load`; the risk is documented
  above, and `pySDC.helpers.fieldsIO` provides a format that does not need pickles.
- Vulnerable dependencies: Dependabot reports known vulnerabilities in dependencies, and every release comes with SBOMs
  (see [Dependencies](#dependencies)) so that installations can be checked against new advisories.
- Tampered releases: releases are built and uploaded by GitHub Actions with signed attestations (see
  [Verifying Releases](#verifying-releases)).
- Unreviewed changes: changes are proposed through pull requests, which run the full test suite and the linters
  `black` and `ruff` and are reviewed before merging, as described in the [governance](./GOVERNANCE.md) and the [CI guide](./docs/contrib/02_continuous_integration.md).

## Comments on This Policy

If you have suggestions on how this process could be improved, please submit a pull request or open an issue to discuss.
