# Automated Test Failure Analysis

**Generated:** 2026-04-20T06:10:44.796786+00:00
**Workflow Run:** https://github.com/Parallel-in-Time/pySDC/actions/runs/24650823394

## Summary

- Total Jobs: 30
- Failed Jobs: 1

## Failed Jobs

### 1. user_firedrake_tests

- **Job ID:** 72073047322
- **Started:** 2026-04-20T05:56:07Z
- **Completed:** 2026-04-20T05:57:24Z
- **Logs:** [View Job Logs](https://github.com/Parallel-in-Time/pySDC/actions/runs/24650823394/job/72073047322)

#### Error Details

**Error 1:**
```
2026-04-20T05:57:20.5005217Z Traceback (most recent call last):
2026-04-20T05:57:20.5005574Z   File "<string>", line 1, in <module>
```

**Error 2:**
```
2026-04-20T05:57:20.5014550Z   File "/repositories/gusto_repo/gusto/recovery/averaging.py", line 11, in <module>
2026-04-20T05:57:20.5014992Z     from firedrake.utils import cached_property
2026-04-20T05:57:20.5015517Z ImportError: cannot import name 'cached_property' from 'firedrake.utils' (/opt/firedrake/firedrake/utils.py)
2026-04-20T05:57:20.8899502Z WARNING! There are options you set that were not used!
2026-04-20T05:57:20.8900627Z WARNING! could be spelling mistake, etc!
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
