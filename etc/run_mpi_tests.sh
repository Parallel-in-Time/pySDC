#!/bin/bash
#
# Run a test tree, giving the MPI tests the ranks they ask for.
#
# `mpi-pytest` runs pytest itself under `mpiexec` and hands each `@pytest.mark.parallel(n)` test
# MPI.COMM_WORLD of a right-sized job, so the suite needs one pytest launch per rank count in use
# rather than one `mpirun` per test case.
#
# The rank counts come from the tests, not from a list kept here or in the workflow: a test asking
# for a count nobody launches would be deselected by every pass and silently never run. Discovering
# them removes that possibility rather than checking for it afterwards.
#
# Every test is covered exactly once. `parallel[n]` claims the tests declaring n ranks, and
# `parallel[1]` claims everything unmarked or explicitly serial, so the serial pass always runs even
# when no test declares MPI at all.
#
# Usage: run_mpi_tests.sh <test-path> [marker]
# Set $PYTEST to override how pytest is invoked (the firedrake job needs `python -m coverage`).
set -u

tests=$1
marker=${2:-}

: "${PYTEST:=coverage run -m pytest --continue-on-collection-errors -v --durations=0}"

marker_args=()
[ -n "$marker" ] && marker_args=(-m "$marker")

ranks=$(PYTHONPATH="$(dirname "$0")${PYTHONPATH:+:$PYTHONPATH}" \
    python -m pytest --collect-only -q -p mpi_ranks "${marker_args[@]}" "$tests" |
    sed -n 's/^MPI_RANKS //p')

echo "Rank counts declared by the tests: ${ranks:-<none>}"

# pytest exits 5 for an empty selection, which is expected whenever a pass has nothing to claim
run_pass() {
    "$@" && return 0
    rc=$?
    [ "$rc" -eq 5 ] && return 0
    return $rc
}

for n in $ranks; do
    [ "$n" -eq 1 ] && continue
    sel="parallel[$n]"
    [ -n "$marker" ] && sel="$marker and $sel"
    run_pass mpiexec -n "$n" $PYTEST -m "$sel" "$tests" || exit $?
done

sel="parallel[1]"
[ -n "$marker" ] && sel="$marker and $sel"
run_pass $PYTEST -m "$sel" "$tests" || exit $?
