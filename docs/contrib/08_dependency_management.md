# Dependency Management in pySDC

## Overview

pySDC uses multiple dependency specification files across the repository:
- **`pyproject.toml`**: Main package dependencies for pip installation
- **`etc/environment-*.yml`**: Environment files for different test configurations
- **`pySDC/projects/*/environment.yml`**: Project-specific environment files

## The Challenge: Loose Constraints and Weekly CI

The repository currently uses loose version constraints (e.g., `numpy>=1.15.4`, `scipy>=0.17.1`) which provides flexibility but can lead to issues:

1. **Unpredictable CI Failures**: Dependencies can update at any time, potentially breaking the weekly Monday morning CI runs
2. **Inconsistent Environments**: Different developers may have different package versions
3. **Debugging Difficulty**: Hard to reproduce issues when dependency versions vary

## Recommended Solutions

### Option 1: Automated Lock Files (Recommended for pySDC)

**Strategy**: Use automated lock file generation to manage exact versions while keeping source files flexible.

**How it works:**
1. **Source files** (`pyproject.toml`, `environment.yml`) keep loose constraints with upper bounds: `numpy>=1.15.4,<3.0`
2. **Lock files** are automatically generated weekly with exact versions
3. **CI uses lock files** for reproducible builds
4. **Developers can use either** source files (flexible) or lock files (reproducible)

**Implementation:**

pySDC now includes automated lock file management:

```bash
# Install tools (one-time setup)
pip install conda-lock pip-tools

# Generate lock files manually
./etc/scripts/update_all_lockfiles.sh

# Or let the automated workflow handle it (runs weekly on Sundays)
```

The automated workflow (`.github/workflows/update_lockfiles.yml`):
- Runs weekly before Monday CI (Sunday 2 AM UTC)
- Regenerates all lock files with latest compatible versions  
- Creates a PR if changes detected
- CI tests the new lock files
- Merge if tests pass

**Lock files created:**
- `etc/lockfiles/environment-*-lock.yml` - For base environments
- `pySDC/projects/*/lockfiles/environment-lock.yml` - For each project
- `requirements-lock.txt` - For pip dependencies

**CI can use lock files** (optional migration):
```yaml
# Install from lock file instead of environment.yml
- name: Install from lock file
  run: micromamba install -f etc/lockfiles/environment-base-lock.yml
```

**Benefits:**
- ✅ Fully automated - no manual constraint updates needed
- ✅ Reproducible builds from lock files
- ✅ Flexible development from source files  
- ✅ Weekly updates keep dependencies current
- ✅ CI catches issues before they reach production
- ✅ Lock files show exactly what's being tested

**Drawbacks:**
- Requires maintaining lock files (automated via workflow)
- Lock files are platform-specific (we use linux-64 for CI)

**See also:** [Automated Lock File README](../../etc/README.md) for implementation details.

### Option 2: Upper Bound Constraints (Current Baseline)

**Strategy**: Add upper bounds to critical dependencies to prevent unexpected major version updates.

**Example** (in `pyproject.toml` and environment files):
```yaml
dependencies:
  - numpy>=1.15.4,<3.0
  - scipy>=0.17.1,<2.0
  - matplotlib>=3.0,<4.0
```

**Benefits**:
- Prevents breaking changes from major version updates
- Simple to implement
- No additional files to maintain

**Drawbacks**:
- Requires knowledge of semantic versioning practices of each dependency
- May block beneficial updates
- Still allows minor/patch updates that could break things

### Option 3: Exact Version Pinning (Not Recommended)

**Strategy**: Pin exact versions for all dependencies.

**Example**:
```yaml
dependencies:
  - numpy==1.26.4
  - scipy==1.11.4
```

**Benefits**:
- Maximum reproducibility
- No unexpected updates

**Drawbacks**:
- Very rigid, hard to maintain
- Prevents security updates
- Can cause dependency conflicts
- Not suitable for a library

### Option 4: Dependabot with Regular Review (Current Approach)

**Strategy**: Use Dependabot to create PRs for dependency updates, review and test before merging.

**Configuration** (see `.github/dependabot.yml`):
```yaml
version: 2
updates:
  - package-ecosystem: pip
    directory: "/"
    schedule:
      interval: daily
    open-pull-requests-limit: 10
```

**Benefits**:
- Automated notifications of updates
- Changes are tested in CI before merging
- Keeps dependencies current

**Drawbacks**:
- Requires active maintenance
- Can create many PRs
- Limited to pip dependencies (doesn't cover conda packages in environment.yml files)

## Recommended Hybrid Approach

For pySDC, we use a combination of strategies:

### 1. Source Files (pyproject.toml, environment.yml)
- **Keep loose lower bounds** for flexibility: `numpy>=1.15.4`
- **Add conservative upper bounds** for major versions: `numpy>=1.15.4,<3.0`
- **Review bounds annually** or when major dependency releases occur
- These serve as the "specification" of what versions are supported

### 2. Automated Lock Files (New!)
- **Weekly automated updates** via GitHub Actions workflow
- **Lock files track exact versions** that pass CI tests
- **Developers choose**: use source files (flexible) or lock files (reproducible)
- **Optional CI migration**: Can switch CI to use lock files for 100% reproducibility

### 3. CI Testing
- **Currently uses source files** with upper-bounded constraints
- **Can optionally use lock files** for exact version reproducibility
- **Weekly scheduled runs** on Monday catch issues early
- **Automated failure handling** creates PRs when issues occur

### 4. Project Environments
- **Maintain project-specific environment.yml** files with the same strategy
- **Lock files auto-generated** for each project
- **Document any special version requirements** in project README files
- **Test projects independently** in CI (already done)

### 5. Developer Workflow
- **Use source files** for normal development (flexible, easy to work with)
- **Use lock files** when exact reproducibility needed (debugging CI issues, etc.)
- **Lock files update automatically** - no manual maintenance required
- **Scripts available** for manual lock file generation if needed

## Implementation Guidelines

### For Main Dependencies (pyproject.toml)

```toml
dependencies = [
    'numpy>=1.15.4,<3.0',
    'scipy>=0.17.1,<2.0', 
    'matplotlib>=3.0,<4.0',
    'sympy>=1.0,<2.0',
    'numba>=0.35,<1.0',
    'dill>=0.2.6',
    'qmat>=0.1.19',
]
```

### For Environment Files

```yaml
name: pySDC
channels:
  - conda-forge
dependencies:
  - numpy>=1.15.4,<3.0
  - scipy>=0.17.1,<2.0
  - matplotlib>=3.0,<4.0
  - dill>=0.2.6
  - pip
  - pip:
    - qmat>=0.1.8
```

### Handling Weekly CI Failures

When the Monday morning CI run fails due to dependency issues:

1. **Check for automated lock file PR**: The Sunday workflow may have already identified the issue
2. **Review the automated failure PR**: Created by the `auto_fix_failures.yml` workflow
3. **Identify the cause**:
   - New dependency version broke compatibility
   - Lock file needs updating
   - Code needs updating for new dependency version
4. **Fix the issue**:
   - If a lock file update PR exists and looks good, merge it
   - If not, manually add an upper bound to the problematic dependency
   - Or update pySDC code to work with the new dependency version
5. **Update lock files**: Run `./etc/scripts/update_all_lockfiles.sh` if constraints changed
6. **Open an issue** to track updating code for the new dependency version (if applicable)

## Monitoring and Maintenance

### Regular Tasks

- **Weekly**: Review automated failure PRs from Monday CI runs
- **Monthly**: Check for security updates to dependencies
- **Quarterly**: Review and update upper bounds as needed
- **Annually**: Audit all dependencies and remove outdated constraints

### Tools

- **GitHub Actions**: Automated weekly testing (Monday 5:01 AM UTC)
- **Auto-fix workflow**: Automatic PR creation for failures
- **Dependabot**: Tracks pip dependencies (configured in `.github/dependabot.yml`)

## Adding New Dependencies

When adding a new dependency to pySDC:

1. **Check compatibility**: Ensure it works with existing dependencies
2. **Add version constraints**: Include both lower and upper bounds
3. **Update all relevant files**:
   - `pyproject.toml` for core dependencies
   - Appropriate `etc/environment-*.yml` files
   - Project-specific `environment.yml` if needed
4. **Test thoroughly**: Run tests locally and in CI
5. **Document**: Note any special requirements or version constraints

## Security Considerations

- **Monitor security advisories**: Check dependencies for known vulnerabilities
- **Apply security patches promptly**: Even if it means updating to a new minor version
- **Test after security updates**: Ensure patches don't break functionality
- **Report vulnerabilities**: Follow the [security policy](../../SECURITY.md)

## For Project Maintainers

When creating a new project in `pySDC/projects/`:

1. **Create `environment.yml`**: Include all project-specific dependencies
2. **Follow the same versioning strategy**: Use lower bounds and conservative upper bounds
3. **Keep it minimal**: Only include dependencies not in the base environment
4. **Document special requirements**: Note any version-specific needs in the project README
5. **Test independently**: Ensure the project environment works in isolation

See [Adding a Project](./06_new_project.md) for more details.

### Project-Specific Constraints

Some projects have **tighter version constraints** than the general guidelines due to specific requirements:

**Example: DAE Project**
```yaml
dependencies:
  - scipy>=0.17.1,<1.15  # Tighter than general <2.0 constraint
```

**Reason**: The DAE project contains tests for fully implicit index-2 differential-algebraic equation solvers that are sensitive to numerical precision changes. Scipy versions >= 1.15 introduce minor numerical differences that cause test failures in convergence order tests.

**When to use tighter constraints**:
- Numerical precision requirements in scientific computing
- Known compatibility issues with specific version ranges
- Project relies on deprecated features being removed in newer versions

**Best practices**:
- Document the reason in the project README or environment file
- Add a comment explaining why the constraint is tighter
- Periodically review if the constraint can be relaxed
- Consider opening an issue to track updating code for newer versions

## Examples

### Good Dependency Specification

```yaml
# environment.yml for a project using MPI
name: pySDC
channels:
  - conda-forge
dependencies:
  - numpy>=1.15.4,<3.0
  - scipy>=0.17.1,<2.0
  - matplotlib>=3.0,<4.0
  - dill>=0.2.6
  - mpich>=3.0,<5.0
  - mpi4py>=3.0.0,<4.0
  - pip
  - pip:
    - qmat>=0.1.8
```

**Note**: Some projects may require tighter constraints due to numerical precision requirements. For example, the DAE project uses `scipy>=0.17.1,<1.15` because newer scipy versions introduce numerical differences that affect the fully implicit index-2 DAE solver tests. Always respect project-specific constraints when they exist.

### Version Constraint Patterns

- **Stable, mature libraries**: `numpy>=1.15.4,<3.0` (allow minor updates within major version)
- **Rapidly evolving libraries**: `pytorch>=2.0,<2.2` (tighter constraints)
- **Internal dependencies**: `qmat>=0.1.8` (trust semantic versioning)
- **System packages**: `mpich>=3.0` (broader range acceptable)

## Troubleshooting

### "Package X conflicts with requirement Y"

**Solution**: Adjust upper/lower bounds to find compatible versions. Use `conda/mamba` solver to identify compatible versions:
```bash
mamba create -n test-env --dry-run -c conda-forge package1 package2
```

### "Tests pass locally but fail in CI"

**Possible causes**:
- Different dependency versions
- Environment file not updated
- CI using newer package versions

**Solution**: 
1. Check CI logs for actual versions installed
2. Try reproducing with same versions locally
3. Update environment files or add constraints

### "Dependabot PRs keep failing"

**Solution**: 
- Review if the dependency update is breaking
- Check if upper bounds need adjustment
- Consider if pySDC code needs updates for new versions

## References

- [Continuous Integration Documentation](./02_continuous_integration.md)
- [Adding a New Project](./06_new_project.md)
- [Python Packaging Guide](https://packaging.python.org/en/latest/)
- [Conda Environment Files](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
- [Semantic Versioning](https://semver.org/)

---

:arrow_left: [Back to Release Guide](./07_release_guide.md) ---
:arrow_up: [Contributing Summary](./../../CONTRIBUTING.md)
