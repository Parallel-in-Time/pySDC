# Continuous Integration in pySDC

Any commit in `pySDC` are tested by GitHub continuous integration (CI). You can see in in the [action panel](https://github.com/Parallel-in-Time/pySDC/actions) the tests for each branches.
Those tests are currently divided in three main categories : [code linting](#code-linting), [code testing](#code-testing) and [code coverage](#code-coverage).
Finally, the CI also build artifacts that are used to generate the documentation website (see http://parallel-in-time.org/pySDC/), more details given in the [documentation generation](#documentation-generation) section.

## CI Workflow Triggers

The CI pipeline is configured to avoid redundant runs. The workflow is triggered:

- **On pull requests**: CI runs for all pull request events (opened, synchronized, reopened, etc.)
- **On push to master**: CI runs when code is pushed directly to the `master` branch
- **On schedule**: CI runs weekly on Monday at 5:01 AM UTC (via cron schedule)

This configuration ensures that when you push commits to a pull request branch, the CI runs only once for the pull request event, not twice (once for the PR and once for the push). Direct pushes to the master branch will still trigger the CI to ensure the main branch is always tested.

### Automated Failure Handling

When the weekly scheduled CI run fails, an automated workflow is triggered to help investigate and resolve the issues:

1. **Automatic Detection**: The `auto_fix_failures.yml` workflow detects failures from the scheduled Monday morning runs
2. **Failure Analysis**: The system analyzes failed jobs and extracts:
   - Error messages and stack traces
   - Links to failed workflow runs and job logs
   - Summary of all failed jobs
3. **Pull Request Creation**: A PR is automatically created with:
   - Detailed failure analysis report
   - Recommendations for investigation
   - Instructions for applying fixes
   - Labels: `automated`, `test-failure`, `needs-investigation`

**How to Handle Automated Failure PRs:**

When you receive an automated failure PR:

1. Review the `failure_analysis.md` file attached to the PR
2. Check the linked workflow run and job logs for full details
3. Investigate the root cause (dependency issues, flaky tests, code bugs, etc.)
4. Push fixes directly to the PR branch or close if it's a transient failure
5. Test fixes locally or wait for CI to verify
6. Merge when the issue is confirmed resolved

For more details, see the [automated failure handling documentation](../../.github/scripts/README.md).

> :bell: **Note:** These automated PRs are informational and require manual review. They help centralize failure information but don't automatically fix issues. If you can identify and fix the problem, push commits to the auto-generated branch.

## Code linting

Code style linting is performed using [black](https://black.readthedocs.io/en/stable/) and [ruff](https://docs.astral.sh/ruff/) for code syntax checking. In particular, `black` is used to check compliance with (most of) [PEP-8 guidelines](https://peps.python.org/pep-0008/).

Those tests are conducted for each commit (even for forks), but you can also run it locally in the root folder of `pySDC` before pushing any commit :

```bash
# Install required packages (works also with conda/mamba)
pip install black ruff
# First : test code style linting with black
black pySDC --check --diff --color
# Second : test code syntax with ruff
ruff check pySDC
```

> :bell: To avoid any error about formatting (`black`), you can simply use this program to reformat your code directly using the command :
>
> ```bash
> black pySDC
> ```

Some style rules that are automatically enforced :

- lines should be not longer than 120 characters
- arithmetic operators (`+`, `*`, ...) should be separated with variables by one empty space

You can automate linting somewhat by using git hooks.
In order to run black automatically, we want to do a pre-commit hook which adds the modified files to the commit after reformatting.
To this end, just add the following to a possibly new file in the path `<pySDC-root-directory>/.git/hooks/pre-commit`:

```bash
#!/bin/sh

# get files that have been staged
export files=$(git diff --staged --name-only HEAD | grep .py | sed -e "s,^,$(git rev-parse --show-toplevel)/,")

# remove any deleted files because black will otherwise fail
for file in $files
do
        if [ ! -f "$file" ]; then
            files=( ${files[@]/$file} )
        fi
done

# apply black and stage the changes that black made
if [[ $files != "" ]]
then
        black $files
        git add $files
fi
```

You may need to run `chmod +x` on the file to allow it to be executed.
Be aware that the hook will alter files you may have opened in an editor whenever you make a commit, which may confuse you(r editor).

To automate ruff, we want to write a hook that alters the commit message in case any errors are detected. This gives us the choice of aborting the commit and fixing the issues, or we can go ahead and commit them and worry about ruff only when the time comes to do a pull request.
To obtain this functionality, add the following to `<pySDC-root-directory>/.git/hooks/prepare-commit-msg`:

```bash
#!/bin/sh

COMMIT_MSG_FILE=$1

export files=$(git diff --staged --name-only HEAD | grep .py | sed -e "s,^,$(git rev-parse --show-toplevel)/,")

if [[ $files != "" ]]
then
        export ruff_output=$(ruff check --quiet $files)
        if [[ "$ruff_output" != "" ]]
        then
                git interpret-trailers --in-place --trailer "$(echo "$ruff_output" | sed -e 's/^/#/')" "$COMMIT_MSG_FILE"
                git interpret-trailers --in-place --trailer "#!!!!!!!!!! WARNING: RUFF FAILED !!!!!!!!!!" "$COMMIT_MSG_FILE"
        fi
fi

```
Don't forget to assign execution rights.

### Using pre-commit hooks (Recommended)

For a more robust and automated approach, you can use the [pre-commit](https://pre-commit.com/) framework, which manages git hooks for you. The repository includes a `.pre-commit-config.yaml` file that is configured to run the exact same linting checks as the CI pipeline.

To set up pre-commit hooks:

```bash
# Install pre-commit (can also be done with conda/mamba or as part of pip install -e .[dev])
pip install pre-commit

# Install the git hooks
pre-commit install

# (Optional) Run against all files to check current state
pre-commit run --all-files
```

Once installed, the hooks will automatically run `black` and `ruff` on staged files before each commit. The hooks are configured to check only files in the `pySDC` directory, matching exactly what the CI pipeline checks.

> :bell: **Note:** The pre-commit hooks run in `--check` mode by default (same as CI), meaning they will prevent commits if formatting issues are found. You can fix formatting issues by running `black pySDC` directly, then staging and committing again.

As a final note, make sure to regularly update linting related packages, as they constantly introduce checking of more PEP8 guidelines.
This might cause the linting to fail in the GitHub action, which uses the most up to date versions available on the conda-forge channel, even though it passed locally.

## Code testing

This is done using [pytest](https://docs.pytest.org/en/7.2.x/), and runs all the tests written in the `pySDC/tests` folder. You can run those locally in the root folder of `pySDC` using :

```bash
# Install required packages (works also with conda/mamba)
pip install pytest pytest-benchmark pytest-timeout coverage[toml]
# Run tests
pytest -v pySDC/tests
```

> :bell: Many components are tested (core, implementations, projects, tutorials, etc ...) which make the testing quite long.
> When working on a single part of the code, you can run only the corresponding part of the test by specifying the test path, for instance :
>
> ```bash
> pytest -v pySDC/tests/test_nodes.py  # only test nodes generation
> ```
>
> You can also run one specific test only like this:
>
> ```bash
> pytest -v pySDC/tests/test_nodes.py::test_nodesGeneration             # only test_nodesGeneration function
> pytest -v pySDC/tests/test_nodes.py::test_nodesGeneration[LEGENDRE]   # only test_nodesGeneration with LEGENDRE nodes
> ```

## Running CI on HPC from pull requests

> :warning: **Note:** The GitLab mirror integration is currently disabled due to technical issues. This section describes functionality that is temporarily unavailable.

By syncing the GitHub repository to a certain Gitlab instance, CI-Jobs can be run on HPC machines. This can be helpful for benchmarks or when running on accelerators that are not available as GitHub runners.

For security and accounting reasons, a few extra steps are needed in order to run the contents of a pull request on HPC:

- The pull request needs to have the tag "gitlab-mirror" assigned to it.
- A person with write-permission for the Parallel-in-Time pySDC repository needs to trigger the workflow. Ask for someone with the required permissions to rerun the workflow if needed.
- The workflow checks if the code can be merged. If this is not the case, the code is not mirrored and the workflow fails. In this case, please merge upstream changes, fix all conflicts, and rerun the workflow.

> :bell: Note that direct pushes to Parallel-in-Time/pySDC will always trigger the HPC pipeline on Gitlab

Regardless of why the Gitlab pipeline was triggered, the following holds true:

- The return-state from Gitlab is transmitted to GitHub (Success/Failure) leading to the same result in GitHub
- Logs from Gitlab are also transferred. The full logs of all jobs can be read from within GitHub. For better overview, these are folded, so unfolding is needed before reading.
- Artifacts from Gitlab jobs are also transferred back to GitHub
- Information, such as coverage is transferred to GitHub, but not yet merged across multiple GitHub workflows. Therefore, there is no complete summary of e.g. coverage-reports across all jobs in all workflows.

> :warning: The coverage report from the HPC tests is not yet merged with other reports. The test coverage will not show up on the respective website or in the badge. We are working on this.

### HPC test environments

In order to run tests on GPUs, please use the pytest marker `cupy`.

If you want to create a new HPC test environment, the following steps need to be completed:

- Create a new slurm job-script in `etc/juwels_*.sh`. The name and location of the file is important.
- Adapt `.gitlab-ci.yml` to include the new job-script. For this, add a name in the job "test_JUWELS" in the section `parallel: matrix: SHELL_SCRIPT`. The name there must match the name of the newly created file.
As a starting point it is recommended to copy and adapt an existing file (e.g. `etc/juwels_cupy.sh`).

## Code coverage

This stage allows to checks how much of the `pySDC` code is tested by the previous stage. It is based on the [coverage](https://pypi.org/project/coverage/) library and currently applied to the following directories :

- `pySDC/core`
- `pySDC/projects`
- `pySDC/tutorial`

This analysis is done in parallel to the test each time a pull is done on any branch (main repository or fork).

### Viewing Coverage Reports

There are multiple ways to view test coverage for pySDC:

1. **HTML Coverage Report** (recommended for detailed analysis)
   - View the current [coverage report for the master branch](https://parallel-in-time.org/pySDC/coverage/index.html)
   - Shows line-by-line coverage with color coding
   - Updated automatically with each push to master

2. **Codecov Dashboard** (for trends and PR analysis)
   - [Compare results with previous builds](https://app.codecov.io/gh/Parallel-in-Time/pySDC)
   - View coverage trends over time
   - See coverage impact of specific commits
   - Codecov automatically comments on pull requests with coverage changes

3. **Coverage Badge** (quick status check)
   - Displayed in the [README](../../README.md)
   - Shows current coverage percentage for master branch

4. **PR Coverage Comments** (for contributors)
   - Codecov comments on each pull request
   - Shows coverage changes introduced by the PR
   - Highlights uncovered lines in modified files

During developments, you can also run the coverage tests locally, using :

```bash
echo "print('Loading sitecustomize.py...');import coverage;coverage.process_startup()" > sitecustomize.py
coverage run -m pytest --continue-on-collection-errors -v --durations=0 pySDC/tests
```

> :bell: Note that this will run all `pySDC` tests while analyzing coverage, hence requires all packages installed for the [code testing stage](#code-testing).

Once the test are finished, you can collect and post-process coverage result :

```bash
coverage combine
python -m coverage html
```

This will generate the coverage report in a `htmlcov` folder, and you can open the `index.html` file within using your favorite browser.

> :warning: Coverage can be lower if some tests fails (for instance, if you did not install all required python package to run all the tests).

### Coverage exceptions

The coverage configuration is defined in two places:

1. **Runtime configuration** (`pyproject.toml`) - Controls what coverage measures during test execution
2. **Codecov configuration** (`codecov.yml`) - Controls how Codecov reports and displays coverage

Some types of code lines will be ignored by the coverage analysis (_e.g_ lines starting with `raise`, ...), see the `[tool.coverage.report]` section in `pyproject.toml`.
Part of code (functions, conditionaly, for loops, etc ...) can be ignored by coverage analysis using the `# pragma: no cover`, for instance

```python
# ...
# code analyzed by coverage
# ...
if condition:  # pragma: no cover
    # code ignored by coverage
# ...
# code analyzed by coverage
# ...
def function():  # pragma: no cover
    # all function code is ignored by coverage
```

Accepted use of the `# pragma: no cover` are:

1. Functions and code used for plotting
2. Lines in one conditional preceding any `raise` statement

If you think the pragma should be used in other parts of your pull request, please indicate (and justify) this in your description.

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
> But you can still generate the website without it: just all images for the tutorials, projects and playgrounds will be missing.
> This approach can be considered for local testing of your contribution when it does not concern parts containing images (_i.e_ project or code documentation).

:arrow_left: [Back to Pull Request Recommendation](./01_pull_requests.md) ---
:arrow_up: [Contributing Summary](./../../CONTRIBUTING.md) ---
:arrow_right: [Next to Naming Conventions](./03_naming_conventions.md)
