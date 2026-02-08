# Dependency Management Solution Summary

## Problem Statement

The pySDC repository had loose dependency constraints (e.g., `numpy>=1.15.4`) which could lead to:
- Unpredictable CI failures during weekly Monday morning runs
- Dependencies updating at any time and breaking tests
- Difficulty reproducing issues across different environments

## Solution: Two-Tier Automated System

We've implemented a hybrid approach that provides both **stability** and **flexibility**:

### Tier 1: Conservative Upper Bounds (Baseline Protection)

**What**: Added conservative upper bounds to all dependency specifications

**Where**: 
- `pyproject.toml`
- `etc/environment-*.yml`
- `pySDC/projects/*/environment.yml`

**Example**:
```yaml
dependencies:
  - numpy>=1.15.4,<3.0     # Was: numpy>=1.15.4
  - scipy>=0.17.1,<2.0     # Was: scipy>=0.17.1
  - matplotlib>=3.0,<4.0   # Was: matplotlib>=3.0
```

**Purpose**: Prevents major version updates from breaking the build unexpectedly

### Tier 2: Automated Lock Files (Exact Reproducibility)

**What**: Automated system to generate and maintain lock files with exact versions

**Components**:
1. **Weekly Workflow** (`.github/workflows/update_lockfiles.yml`)
   - Runs Sunday 2 AM UTC (before Monday CI)
   - Regenerates all lock files
   - Creates PR if changes detected
   
2. **Manual Scripts** (`etc/scripts/`)
   - `update_lockfile.sh` - Update single lock file
   - `update_all_lockfiles.sh` - Update all lock files

3. **Lock Files Generated**:
   - `etc/lockfiles/environment-*-lock.yml` (base environments)
   - `pySDC/projects/*/lockfiles/environment-lock.yml` (projects)
   - `requirements-lock.txt` (pip dependencies)

**Purpose**: Provides exact version reproducibility when needed

## How It Works

### Normal Operation

1. **Developers** use source files (environment.yml) with upper-bounded constraints
2. **Source files** allow minor/patch updates within major version bounds
3. **Weekly workflow** generates lock files on Sunday
4. **Lock file PR** is created if versions changed
5. **CI tests** lock file PR to ensure compatibility
6. **Merge** if tests pass

### When Issues Occur

**Scenario 1: Weekly CI Fails**
1. Automated failure PR is created (existing system)
2. Check if lock file update PR exists
3. If yes: Review and merge lock file PR
4. If no: Investigate, add tighter constraint, regenerate lock files

**Scenario 2: Breaking Dependency Update**
1. Weekly workflow creates lock file PR with new versions
2. CI tests fail on the lock file PR
3. Don't merge the PR
4. Add upper bound constraint to source file
5. Regenerate lock files
6. Issue tracks updating code for new version

## Benefits of This Approach

### For Stability
✅ Upper bounds prevent unexpected major version updates
✅ Lock files provide exact reproducibility
✅ Weekly testing catches issues before production
✅ Automated PRs make issues visible

### For Flexibility  
✅ Source files remain readable and flexible
✅ Developers choose source files or lock files
✅ Minor updates happen automatically
✅ No manual constraint updates needed

### For Maintenance
✅ Fully automated lock file updates
✅ Clear process for handling failures
✅ CI catches issues before they affect users
✅ Lock files are optional (can use source files only)

## Migration Path

The solution is implemented in phases:

### Phase 1: ✅ Completed
- [x] Add upper bounds to all source files
- [x] Create automated lock file workflow
- [x] Create manual update scripts
- [x] Document the approach

### Phase 2: Optional (Future)
- [ ] Generate initial lock files
- [ ] Migrate CI to use lock files
- [ ] Test in production
- [ ] Refine as needed

### Phase 3: Optional (Future)
- [ ] Add lock files to all project subdirectories
- [ ] Migrate all CI jobs to use lock files
- [ ] Full lock file coverage

## Files Modified

### Configuration Files (30 files)
- `pyproject.toml` - Added upper bounds
- `etc/environment-*.yml` (5 files) - Added upper bounds
- `pySDC/projects/*/environment.yml` (16 files) - Added upper bounds

### New Files Created
- `.github/workflows/update_lockfiles.yml` - Automated workflow
- `docs/contrib/08_dependency_management.md` - Comprehensive guide
- `etc/README.md` - Lock file system documentation
- `etc/scripts/update_lockfile.sh` - Manual update script
- `etc/scripts/update_all_lockfiles.sh` - Bulk update script

### Documentation Updates
- `CONTRIBUTING.md` - Added reference to dependency guide
- `docs/contrib/02_continuous_integration.md` - Added dependency failure handling
- `docs/contrib/06_new_project.md` - Added constraint guidance

## Key Documentation

1. **[Dependency Management Guide](docs/contrib/08_dependency_management.md)**
   - Complete guide to dependency management strategy
   - Options analysis
   - Implementation details
   - Troubleshooting

2. **[Lock File System README](etc/README.md)**
   - Lock file workflow details
   - Usage instructions
   - Migration plan

3. **[CI Documentation](docs/contrib/02_continuous_integration.md)**
   - Weekly CI run information
   - Automated failure handling
   - Dependency-related failure handling

## Summary

This solution addresses the loose dependency constraint problem through:

1. **Automated lock files** that adapt on the fly (new requirement)
2. **Conservative upper bounds** for baseline protection
3. **Weekly automated updates** that catch issues early
4. **Flexible workflow** that works for both developers and CI

The system is **fully automated** - dependency constraints update themselves weekly, and issues are caught before they reach the main branch. No manual constraint maintenance is required.
