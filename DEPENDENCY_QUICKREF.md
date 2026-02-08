# Quick Reference: Dependency Management

## For Users

**Installing pySDC:**
```bash
# Using pip (gets core dependencies)
pip install pySDC

# Using conda with environment file (recommended for development)
git clone https://github.com/Parallel-in-Time/pySDC.git
cd pySDC
micromamba env create -f etc/environment-base.yml
micromamba activate pySDC
pip install --no-deps -e .
```

**Using lock files for exact reproducibility:**
```bash
# If lock files are available (coming soon)
micromamba install -f etc/lockfiles/environment-base-lock.yml
```

## For Contributors

**Adding dependencies to your project:**
1. Edit `pySDC/projects/YOUR_PROJECT/environment.yml`
2. Use format: `package>=min_version,<max_major_version`
3. Example: `numpy>=1.20.0,<3.0`

See: [Adding a Project Guide](docs/contrib/06_new_project.md)

**Updating dependencies:**
- Lock files update automatically weekly (Sundays)
- You don't need to manually update version constraints
- Just specify what you need with reasonable bounds

## For Maintainers

**When lock file PR is created (Sundays):**
1. Review the PR created by the automated workflow
2. Check CI results
3. If tests pass: Merge
4. If tests fail: Investigate, may need to tighten constraints

**When Monday CI fails:**
1. Check if a lock file update PR exists
2. Review automated failure analysis PR
3. Identify problematic dependency
4. Options:
   - Merge lock file PR if it fixes the issue
   - Add tighter upper bound in source file
   - Update code for new dependency version

**Manual lock file update:**
```bash
# Update all lock files
./etc/scripts/update_all_lockfiles.sh

# Update specific environment
./etc/scripts/update_lockfile.sh etc/environment-base.yml
```

**Adding upper bound to dependency:**
1. Edit the relevant environment.yml or pyproject.toml
2. Change `package>=X.Y` to `package>=X.Y,<Z.0`
3. Regenerate lock files: `./etc/scripts/update_all_lockfiles.sh`
4. Commit changes

## Common Tasks

### I want reproducible builds
**Use lock files** (when available):
```bash
micromamba install -f etc/lockfiles/environment-base-lock.yml
```

### I want the latest versions
**Use source files**:
```bash
micromamba env create -f etc/environment-base.yml
```

### I need to pin a specific version
**Edit the environment.yml**:
```yaml
dependencies:
  - package==X.Y.Z  # Exact version
```
Then update lock files.

### A dependency broke my code
**Add upper bound**:
```yaml
dependencies:
  - package>=X.Y,<X.Z  # Block the breaking version
```
Then update lock files and open issue to fix code.

## Key Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Core pip dependencies |
| `etc/environment-*.yml` | Base environments for different configs |
| `pySDC/projects/*/environment.yml` | Project-specific dependencies |
| `etc/lockfiles/*-lock.yml` | Exact versions (auto-generated) |
| `.github/workflows/update_lockfiles.yml` | Automated lock file updates |

## Documentation

- **[Complete Guide](docs/contrib/08_dependency_management.md)** - Full dependency management documentation
- **[Lock File System](etc/README.md)** - Lock file implementation details
- **[Timeline](DEPENDENCY_TIMELINE.md)** - Weekly workflow visualization
- **[Solution Summary](DEPENDENCY_SOLUTION.md)** - Overview of the solution

## Weekly Schedule

- **Sunday 2:00 AM UTC**: Lock files update automatically
- **Monday 5:01 AM UTC**: Weekly CI run

This ensures lock files are tested before the weekly CI run.

## Need Help?

1. Check [Dependency Management Guide](docs/contrib/08_dependency_management.md)
2. Review [Troubleshooting section](docs/contrib/08_dependency_management.md#troubleshooting)
3. Check automated failure PR (if Monday CI failed)
4. Check lock file update PR (if available)
5. Open an issue with details
