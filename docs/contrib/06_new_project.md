# Adding a project to pySDC (and automatic testing)

When you do a project with pySDC we appreciate it if you merge it back to the main repository.
We are committed to keeping your work reproducible and prevent it from fading into oblivion.
To that end, please write extensive tests for your code and add them to the project.
See the [contribution guide](./../../CONTRIBUTING.md) for general advice on testing etc.
This guide will detail only how to add a project to pySDC.

## Add a directory in 'pySDC/projects'

First, create a new directory in `pySDC/projects` with the name of your project.
The code of the new project should go into that newly created directory.

## Add an environment-file

The testing pipeline uses [micromamba](<https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html>)
and requires an environment file for setup.
This includes the dependencies that are required for your project to run.
The file needs to be named `environment.yml` and needs to follow the structure shown below:

```yaml
name: pySDC
channels:
  - conda-forge
  - defaults
dependencies:
  - numpy
```

The list of dependencies can be extended as needed.
The name should stay `pySDC`. The channels cover most of the usual packages.
If a package is needed that cannot be found in those channels by conda (or mamba),
please add the correct channel to the list.

## Add tests to the project

In order to automatically find the tests of your project, please add the tests to a subdirectory called `tests` in the directory of your project.
Furthermore, the files should start with `test_` and the functions should also start with that.
For more information see the documentation of pytest on [test discovery](<https://docs.pytest.org/en/8.2.x/explanation/goodpractices.html#tests-as-part-of-application-code>).

## Add the project to the continuous integration pipeline

To run the tests of all projects in parallel, the projects are explicitly mentioned in the CI-file.
In order to run the tests of your project, please add the name of your project **as your directory is named**
in the [CI-File](<https://github.com/Parallel-in-Time/pySDC/blob/master/.github/workflows/ci_pipeline.yml>)
in the job `project_cpu_tests_linux` in the list `strategy/matrix/env`.

## Getting a DOI of pySDC for publication

If your project is published and you need a dedicated pySDC version with a DOI, please get in touch with us and/or open a new issue.
We will help you with this as soon as possible.
Note that a final DOI is usually only necessary once a paper is accepted and the final proofs are due.
We strongly encourage to describe and cite the current version of pySDC already during initial submission, though.

:arrow_left: [Back to Documenting Code](./05_documenting_code.md) ---
:arrow_up: [Contributing Summary](./../../CONTRIBUTING.md) ---
:arrow_right: [Next to a cute picture of cat](https://www.vecteezy.com/photo/2098203-silver-tabby-cat-sitting-on-green-background)