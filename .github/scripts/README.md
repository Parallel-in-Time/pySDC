# Automated Test Failure Handling

This directory contains scripts and workflows for automatically handling test failures in the pySDC CI pipeline.

## Overview

When the weekly CI tests (scheduled for Monday mornings) fail, the automated workflow will:

1. Detect the failure
2. Analyze the failed jobs and extract error information
3. Create a Pull Request with:
   - A detailed failure analysis report
   - Links to the failed workflow run and job logs
   - Recommended actions for investigation and fixes
   - Instructions on how to apply fixes

## Components

### Workflows

- **`auto_fix_failures.yml`**: Main workflow that triggers on CI pipeline failures
  - Only activates for scheduled (Monday morning) runs that fail
  - Analyzes failures and creates a PR automatically

### Scripts

- **`analyze_failures.py`**: Analyzes workflow run failures
  - Fetches job information from GitHub API
  - Extracts error messages and tracebacks from logs
  - Generates a detailed markdown report
  - Saves analysis as JSON for further processing

- **`create_failure_pr.py`**: Creates a Pull Request for failures
  - Uses the analysis from `analyze_failures.py`
  - Creates a formatted PR with all relevant information
  - Adds appropriate labels for easy identification

## How It Works

1. **Detection**: The `workflow_run` trigger monitors the "CI pipeline for pySDC" workflow
2. **Filtering**: Only runs that failed AND were triggered by schedule (Monday cron job) activate the auto-fix workflow
3. **Analysis**: The workflow checks out the code and runs the analysis script
4. **Reporting**: A new branch is created with the failure analysis
5. **PR Creation**: An automated PR is opened with the analysis and instructions

## Usage

### For Maintainers

When you receive an automated failure PR:

1. Review the `failure_analysis.md` file in the PR
2. Check the linked workflow run and job logs
3. Investigate the root cause of the failures
4. Apply fixes by pushing commits to the PR branch
5. Test your fixes locally or wait for CI to run on the PR
6. Merge when the issue is resolved

### Configuration

The workflow requires the following permissions (already configured):
- `contents: write` - To create branches and commit files
- `pull-requests: write` - To create PRs
- `issues: write` - To add labels
- `actions: read` - To read workflow run information

### Customization

You can customize the behavior by editing:

- **Trigger conditions** in `auto_fix_failures.yml`:
  ```yaml
  if: >-
    ${{ github.event.workflow_run.conclusion == 'failure' 
    && github.event.workflow_run.event == 'schedule' }}
  ```

- **Error patterns** in `analyze_failures.py`:
  ```python
  error_patterns = [
      'ERROR:',
      'FAILED',
      # Add more patterns here
  ]
  ```

- **PR labels** in `create_failure_pr.py`:
  ```python
  labels_data = {'labels': ['automated', 'test-failure', 'needs-investigation']}
  ```

## Example PR Structure

An automated PR will include:

- **Title**: `🔴 Auto-fix: Weekly test failures (run_id)`
- **Body**: Summary of failures, workflow run link, next steps
- **Files**: `failure_analysis.md` with detailed error information
- **Labels**: `automated`, `test-failure`, `needs-investigation`

## Troubleshooting

### The workflow doesn't trigger

- Check that the CI pipeline workflow is named exactly "CI pipeline for pySDC"
- Verify that the run was triggered by schedule (not push/PR)
- Ensure the workflow run actually failed

### No analysis file is created

- Check the workflow logs for the "Analyze test failures" step
- Verify that `GITHUB_TOKEN` has sufficient permissions
- Check if the API rate limit was exceeded

### PR creation fails

- Ensure `GITHUB_TOKEN` has write permissions
- Check if a PR already exists for the branch
- Verify that there are actual changes to commit

## Dependencies

The scripts require:
- Python 3.7+
- `requests` library
- `PyGithub` library (for potential future enhancements)

These are installed automatically in the workflow.

## Future Enhancements

Potential improvements:

- [ ] Automatic fix suggestions using AI/LLM
- [ ] Pattern recognition for common failures
- [ ] Integration with issue tracking
- [ ] Notification to relevant maintainers
- [ ] Automatic retry of flaky tests
- [ ] Historical failure analysis and trends
