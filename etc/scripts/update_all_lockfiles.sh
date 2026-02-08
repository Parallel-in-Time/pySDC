#!/bin/bash
# Update all lock files in the repository
# Usage: ./update_all_lockfiles.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Updating all lock files in $REPO_ROOT"
echo "========================================"

# Check if conda-lock is installed
if ! command -v conda-lock &> /dev/null; then
    echo "Error: conda-lock is not installed"
    echo "Install it with: pip install conda-lock"
    exit 1
fi

# Update etc environment lock files
echo ""
echo "Updating etc/ environment lock files..."
mkdir -p "$REPO_ROOT/etc/lockfiles"

for env_file in "$REPO_ROOT"/etc/environment-*.yml; do
    if [ -f "$env_file" ]; then
        echo "  - $(basename "$env_file")"
        "$SCRIPT_DIR/update_lockfile.sh" "$env_file"
    fi
done

# Update project environment lock files
echo ""
echo "Updating project environment lock files..."

for env_file in "$REPO_ROOT"/pySDC/projects/*/environment.yml; do
    if [ -f "$env_file" ]; then
        project_name=$(basename "$(dirname "$env_file")")
        echo "  - $project_name"
        "$SCRIPT_DIR/update_lockfile.sh" "$env_file"
    fi
done

# Update pip lock file
echo ""
echo "Updating pip lock file..."
if command -v pip-compile &> /dev/null; then
    cd "$REPO_ROOT"
    pip-compile pyproject.toml --resolver=backtracking -o requirements-lock.txt
    echo "✅ requirements-lock.txt updated"
else
    echo "⚠️  pip-tools not installed, skipping pip lock file"
    echo "   Install with: pip install pip-tools"
fi

echo ""
echo "========================================"
echo "✅ All lock files updated successfully!"
echo ""
echo "To use these lock files in CI, update the workflow to install from:"
echo "  - etc/lockfiles/environment-*-lock.yml"
echo "  - pySDC/projects/*/lockfiles/environment-lock.yml"
