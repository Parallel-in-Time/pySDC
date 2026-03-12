# Example: Automated Test Failure Analysis

This document shows an example of what the automated failure analysis would look like when the weekly tests fail.

## Scenario

The CI pipeline runs on Monday morning at 5:01 AM UTC (as scheduled by cron: '1 5 * * 1'). One or more tests fail.

## Automated Response

### 1. Failure Detection

The `auto_fix_failures.yml` workflow is triggered via `workflow_run` event when:
- The "CI pipeline for pySDC" workflow completes
- The conclusion is "failure"
- The trigger event was "schedule" (Monday morning run)

### 2. Analysis Process

The workflow:
1. Checks out the repository
2. Installs Python and required dependencies (requests, PyGithub)
3. Runs `analyze_failures.py` which:
   - Fetches all jobs from the failed workflow run via GitHub API
   - Downloads logs for each failed job
   - Extracts error messages, tracebacks, and failure patterns
   - Generates a detailed markdown report
   - Saves both markdown and JSON versions

### 3. Branch and PR Creation

The workflow:
1. Creates a new branch named `auto-fix/test-failure-YYYYMMDD-HHMMSS`
2. Commits the failure analysis files
3. Runs `create_failure_pr.py` which:
   - Creates a pull request from the new branch to master
   - Includes a comprehensive description with links and instructions
   - Adds labels: `automated`, `test-failure`, `needs-investigation`

## Example Output

### Example PR Title
```
🔴 Auto-fix: Weekly test failures (12345678)
```

### Example PR Description
```markdown
## 🔴 Automated Test Failure Report

This PR was automatically created in response to test failures in the weekly CI run.

### Summary
- **Workflow Run:** https://github.com/Parallel-in-Time/pySDC/actions/runs/12345678
- **Failed Jobs:** 3 out of 25
- **Trigger:** Weekly scheduled run (Monday morning)

### What This PR Contains

This PR includes an automated analysis of the test failures. The detailed report can be found in the committed `failure_analysis.md` file.

### Next Steps

1. **Review the Analysis:** Check the `failure_analysis.md` file for detailed error information
2. **Investigate Root Cause:** Review the workflow logs and error messages
3. **Apply Fixes:** If you identify the issue, commit fixes to this branch
4. **Test Locally:** Reproduce and verify the fix before merging
5. **Update CI:** Ensure the fix resolves the weekly test failures

### How to Fix Issues

You can push commits directly to this branch:

```bash
git fetch origin
git checkout auto-fix/test-failure-20240101-050500
# Make your changes
git add .
git commit -m "Fix: describe your fix"
git push origin auto-fix/test-failure-20240101-050500
```

### Alternative Actions

- If this is a **transient failure**, you can close this PR
- If this requires **more investigation**, convert this PR to an issue
- If this is a **known issue**, link it to existing issues/PRs

---

**Note:** This is an automated PR. Please review carefully before merging.
```

### Example `failure_analysis.md` Content

```markdown
# Automated Test Failure Analysis

**Generated:** 2024-01-01T05:15:00Z
**Workflow Run:** https://github.com/Parallel-in-Time/pySDC/actions/runs/12345678

## Summary

- Total Jobs: 25
- Failed Jobs: 3

## Failed Jobs

### 1. user_cpu_tests_linux (base, 3.10)

- **Job ID:** 23456789
- **Started:** 2024-01-01T05:02:00Z
- **Completed:** 2024-01-01T05:10:00Z
- **Logs:** [View Job Logs](https://github.com/Parallel-in-Time/pySDC/actions/runs/12345678/jobs/23456789)

#### Error Details

**Error 1:**
```
FAILED pySDC/tests/test_something.py::test_feature - AssertionError: assert 5 == 6
E       assert 5 == 6
```

**Error 2:**
```
Traceback (most recent call last):
  File "pySDC/core/something.py", line 142, in method
    result = self.compute()
  File "pySDC/core/something.py", line 200, in compute
    value = dependency.get_value()
AttributeError: 'NoneType' object has no attribute 'get_value'
```

### 2. user_cpu_tests_linux (pytorch, 3.13)

- **Job ID:** 23456790
- **Started:** 2024-01-01T05:02:00Z
- **Completed:** 2024-01-01T05:12:00Z
- **Logs:** [View Job Logs](https://github.com/Parallel-in-Time/pySDC/actions/runs/12345678/jobs/23456790)

#### Error Details

**Error 1:**
```
ModuleNotFoundError: No module named 'torch'
ERROR: Could not import pytorch dependencies
```

### 3. project_cpu_tests_linux (RDC)

- **Job ID:** 23456791
- **Started:** 2024-01-01T05:05:00Z
- **Completed:** 2024-01-01T05:14:00Z
- **Logs:** [View Job Logs](https://github.com/Parallel-in-Time/pySDC/actions/runs/12345678/jobs/23456791)

#### Error Details

**Error 1:**
```
ImportError: cannot import name 'RDC_Controller' from 'pySDC.implementations.controllers'
```

## Recommended Actions

1. Review the error messages above
2. Check if this is a known issue in recent commits
3. Review the full logs linked above for complete context
4. Consider if this is related to:
   - Dependency updates (check recent dependency changes)
   - Environment configuration issues
   - Test infrastructure problems
   - Flaky tests that need to be fixed
5. If needed, manually investigate and apply fixes to this PR

## How to Use This PR

This PR was automatically created to help investigate test failures. You can:

- Use this PR to track the investigation
- Add commits with fixes directly to this branch
- Close this PR if the issue is resolved elsewhere
- Convert this to an issue if it needs more discussion
```

## Benefits

1. **Immediate Notification**: Team is notified via PR instead of just email
2. **Centralized Tracking**: All failure information in one place
3. **Actionable**: PR branch can be used to apply fixes directly
4. **Historical Record**: PRs remain in history for future reference
5. **Reduced Manual Work**: No need to manually dig through CI logs
6. **Easy Collaboration**: Team members can comment and contribute

## Workflow Permissions

The workflow uses these permissions:
- `contents: write` - To create branches and commit files
- `pull-requests: write` - To create and label PRs
- `issues: write` - To add labels
- `actions: read` - To read workflow run and job information

All of these use the default `GITHUB_TOKEN`, no additional secrets needed.
