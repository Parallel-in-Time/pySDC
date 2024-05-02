# Adding a project to pySDC (and automatic testing)

When adding a new project to pySDC, it needs to be tested.
In order to run those tests as fast as possible, a few steps are needed, that are given below.

## Add a directory in 'pySDC/projects'

First, create a new directory in `pySDC/projects` with the name of your project.
The code of the new project should go into that newly created directory.

## Add an environment-file

In addition to the codde of your projects, an environment-file is needed.
This includes the dependencies that are required for your project to run.
The file needs to be named `env_update.yml` and needs to follow the structure shown below:

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

In order to automatically find the tests of your project, please add the tests to a subdirectory called `tests` in the directory of your project. Furthermore, the files should start with `test_` and the functions should also start with that. For more information see the documentation of pytest on [test discovery](<https://docs.pytest.org/en/8.2.x/explanation/goodpractices.html#tests-as-part-of-application-code>).

## Add the project in the CI-file

To run the tests of all projects in parallel, the projects are explicitly mentioned in the CI-file.
In order to run the tests of your project, please add the name of your project **as your directory is named**
in the [CI-File](./../../.github/workflows/ci_pipeline.yml)
in the job `project_cpu_tests_linux` in the list `strategy/matrix/env`.
