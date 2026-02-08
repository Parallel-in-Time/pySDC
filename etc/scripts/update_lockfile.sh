#!/bin/bash
# Update a single lock file from an environment.yml file
# Usage: ./update_lockfile.sh path/to/environment.yml

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <path-to-environment.yml>"
    echo "Example: $0 etc/environment-base.yml"
    exit 1
fi

ENV_FILE="$1"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: Environment file not found: $ENV_FILE"
    exit 1
fi

# Check if conda-lock is installed
if ! command -v conda-lock &> /dev/null; then
    echo "Error: conda-lock is not installed"
    echo "Install it with: pip install conda-lock"
    exit 1
fi

# Determine output location
DIR=$(dirname "$ENV_FILE")
BASE=$(basename "$ENV_FILE" .yml)

# Create lockfiles directory
if [[ "$ENV_FILE" == etc/* ]]; then
    mkdir -p etc/lockfiles
    LOCK_FILE="etc/lockfiles/${BASE}-lock.yml"
else
    # For project files
    mkdir -p "$DIR/lockfiles"
    LOCK_FILE="$DIR/lockfiles/environment-lock.yml"
fi

echo "Updating lock file for: $ENV_FILE"
echo "Output: $LOCK_FILE"

# Generate lock file
conda-lock lock \
    --file "$ENV_FILE" \
    --platform linux-64 \
    --lockfile "$LOCK_FILE"

echo "✅ Lock file updated successfully: $LOCK_FILE"
