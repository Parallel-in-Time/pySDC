# Automated Dependency Management

This directory contains tools for automated dependency constraint management in pySDC.

## Overview

Instead of manually maintaining version constraints, we use:
1. **Lock files** for reproducible environments
2. **Automated workflows** to update lock files weekly
3. **Source files** keep loose constraints for flexibility

## Lock File Strategy

### For Conda Environments (Recommended)

We use `conda-lock` to generate cross-platform lock files from environment.yml files:

```bash
# Install conda-lock
pip install conda-lock

# Generate lock file for a single environment
conda-lock -f etc/environment-base.yml -p linux-64

# Generate lock files for all platforms
conda-lock -f etc/environment-base.yml -p linux-64 -p osx-64 -p win-64
```

Lock files are committed to the repository and used in CI for reproducible builds.

### For pip (pyproject.toml)

We use `pip-tools` to generate pinned requirements:

```bash
# Install pip-tools
pip install pip-tools

# Generate requirements.txt from pyproject.toml
pip-compile pyproject.toml -o requirements-lock.txt
```

## Automated Workflow

The `.github/workflows/update_lockfiles.yml` workflow:
- Runs weekly (or manually triggered)
- Regenerates all lock files with latest compatible versions
- Creates a PR if lock files changed
- CI tests the new lock files before merging

## Usage in CI

Instead of installing from environment.yml directly:

```yaml
# Old approach (uses loose constraints)
- name: Install environment
  run: micromamba install -f etc/environment-base.yml

# New approach (uses lock file for reproducibility)
- name: Install from lock file
  run: micromamba install -f etc/conda-lock.yml
```

## Benefits

1. **Reproducible builds**: Lock files ensure exact versions
2. **Automatic updates**: Weekly workflow keeps dependencies current
3. **Early detection**: CI catches breaking changes before they reach main branch
4. **Flexible source**: environment.yml keeps loose constraints for developer flexibility
5. **No manual maintenance**: Constraints update automatically

## Directory Structure

```
etc/
  ├── environment-base.yml          # Source with loose constraints
  ├── environment-base-lock.yml     # Generated lock file
  ├── environment-mpi4py.yml        # Source with loose constraints
  ├── environment-mpi4py-lock.yml   # Generated lock file
  └── ...
```

## Manual Lock File Update

To update lock files manually:

```bash
# Update a specific environment
./etc/scripts/update_lockfile.sh etc/environment-base.yml

# Update all environments
./etc/scripts/update_all_lockfiles.sh
```

## When Lock Files Fail in CI

If the Monday CI run fails due to dependency issues:

1. The automated failure PR is created (existing workflow)
2. Check if a lock file update PR exists
3. Review and merge the lock file update PR to fix the issue
4. If no lock file update PR exists, manually investigate and update constraints

## Migration Plan

1. ✅ Keep loose constraints in source files (done)
2. ⬜ Generate initial lock files for all environments
3. ⬜ Update CI to use lock files
4. ⬜ Set up automated lock file update workflow
5. ⬜ Monitor and adjust as needed
