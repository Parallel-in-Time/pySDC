# Weekly Dependency Management Timeline

This document illustrates how the automated dependency management system works throughout the week.

## Weekly Timeline

```
Sunday 2:00 AM UTC
│
├─ Lock File Update Workflow Runs
│  └─ Regenerates all lock files with latest compatible versions
│  └─ Creates PR if changes detected
│
└─ Sunday (rest of day)
   └─ Review lock file update PR (if created)
   └─ CI tests the new versions
   └─ Merge if tests pass

Monday 5:01 AM UTC
│
├─ Weekly CI Run
│  └─ Tests with current dependencies
│  └─ If fails: Automated failure PR created
│  └─ If lock file PR was merged: Tests with updated versions
│
└─ Monday (rest of day)
   └─ Review any failure PRs
   └─ Apply fixes if needed

Tuesday - Saturday
│
└─ Normal Development
   └─ PRs tested with current dependencies
   └─ Lock files stable unless manually updated
```

## Dependency Update Scenarios

### Scenario 1: Compatible Update (Happy Path)

```
Sunday:
  1. NumPy 1.26.3 → 1.26.4 (patch update)
  2. Lock file workflow creates PR
  3. CI tests pass ✅
  4. PR merged automatically or by maintainer

Monday:
  5. Weekly CI runs with NumPy 1.26.4
  6. Tests pass ✅
  7. No action needed
```

### Scenario 2: Breaking Update (Caught Early)

```
Sunday:
  1. SciPy 1.11.4 → 1.14.0 (minor update with breaking change)
  2. Lock file workflow creates PR
  3. CI tests FAIL ❌
  4. PR not merged - issue identified early

Monday:
  5. Weekly CI runs with SciPy 1.11.4 (old version)
  6. Tests pass ✅
  7. Lock file PR remains open for investigation
  
Later:
  8. Maintainer reviews lock file PR
  9. Identifies SciPy 1.14.0 breaking change
  10. Updates upper bound: scipy>=0.17.1,<1.14
  11. Regenerates lock files
  12. Opens issue to track code updates for SciPy 1.14
```

### Scenario 3: Major Version Update (Blocked)

```
Sunday:
  1. NumPy 3.0.0 released (major version)
  2. Lock file workflow runs
  3. NumPy 3.0.0 blocked by upper bound (<3.0) ✅
  4. No lock file changes
  5. No PR created

Monday:
  6. Weekly CI runs normally
  7. Tests pass ✅
  8. System stable
  
Later:
  9. Maintainer decides to support NumPy 3.0
  10. Updates code for NumPy 3.0 compatibility
  11. Changes upper bound to <4.0
  12. Lock files update to NumPy 3.0.x on next Sunday
```

## Two-Layer Protection

### Layer 1: Upper Bounds (Source Files)
```yaml
# pyproject.toml & environment.yml
dependencies:
  - numpy>=1.15.4,<3.0      # Blocks major version 3.x
  - scipy>=0.17.1,<2.0      # Blocks major version 2.x
```

**Purpose**: Prevent unexpected major version updates
**Scope**: All installations (dev, CI, production)

### Layer 2: Lock Files (Generated Weekly)
```yaml
# etc/lockfiles/environment-base-lock.yml
dependencies:
  - numpy==1.26.4           # Exact version
  - scipy==1.11.4           # Exact version
```

**Purpose**: Exact reproducibility when needed
**Scope**: Optional (can be used in CI for 100% reproducibility)

## Benefits Summary

### For Weekly CI Runs
✅ Lock files update Sunday (before Monday run)
✅ Breaking changes caught in lock file PR (not Monday CI)
✅ Monday CI tests known-good versions
✅ No unexpected failures from random dependency updates

### For Developers
✅ Source files remain simple and readable
✅ Can use lock files for reproducibility
✅ Can use source files for flexibility
✅ No manual constraint updates needed

### For Maintainers
✅ Automated PRs show exactly what's changing
✅ Clear decision point (merge lock file PR or not)
✅ Early warning system for breaking changes
✅ Minimal manual intervention required

## Key Configuration Files

- **Source Files**: `pyproject.toml`, `etc/environment-*.yml`, `pySDC/projects/*/environment.yml`
  - Loose constraints with upper bounds
  - Human-readable and maintainable
  
- **Lock Files**: `etc/lockfiles/*-lock.yml`, `pySDC/projects/*/lockfiles/environment-lock.yml`
  - Exact versions
  - Auto-generated weekly
  - Committed to repo

- **Workflows**: 
  - `.github/workflows/update_lockfiles.yml` - Sunday 2 AM UTC
  - `.github/workflows/ci_pipeline.yml` - Monday 5:01 AM UTC
  
- **Scripts**: 
  - `etc/scripts/update_lockfile.sh` - Update single lock file
  - `etc/scripts/update_all_lockfiles.sh` - Update all lock files
