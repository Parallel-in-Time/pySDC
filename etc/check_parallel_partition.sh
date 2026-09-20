#!/bin/bash
#
# Every test a marker selects must be claimed by exactly one of the rank-count passes the workflow
# launches. The rank counts live in the workflow while `@pytest.mark.parallel(n)` lives in the test,
# so a test asking for a count nobody launches is deselected by every pass and silently never runs.
# That is invisible in a green pipeline, which is why it is checked rather than trusted.
#
# Usage: check_parallel_partition.sh <test-path> <marker> <rank counts...>
# The serial pass (`parallel[1]`) is always included; pass the counts the workflow launches beyond it.
#
# Collection alone is enough -- mpi-pytest attaches the `parallel[n]` markers at collection time, so
# this needs no MPI job and costs a few seconds.
set -u

tests=$1
marker=$2
shift 2

# The marker may be empty, for a test tree selected by path rather than by marker.
expr() {
    if [ -z "$marker" ]; then echo "$1"; else echo "$marker and $1"; fi
}

count() {
    python -m pytest --collect-only -q -m "$1" "$tests" 2>/dev/null |
        grep -oE '^[0-9]+/[0-9]+ tests collected|^[0-9]+ tests collected' |
        grep -oE '^[0-9]+' | head -1
}

if [ -z "$marker" ]; then
    total=$(python -m pytest --collect-only -q "$tests" 2>/dev/null |
        grep -oE '^[0-9]+/[0-9]+ tests collected|^[0-9]+ tests collected' |
        grep -oE '^[0-9]+' | head -1)
else
    total=$(count "$marker")
fi
total=${total:-0}

sum=0
for n in 1 "$@"; do
    c=$(count "$(expr "parallel[$n]")")
    c=${c:-0}
    echo "  parallel[$n] : $c"
    sum=$((sum + c))
done

echo "  ---------------"
echo "  sum $sum vs total $total for -m '${marker:-<all>}'"

if [ "$sum" -ne "$total" ]; then
    echo "::error::Rank-count partition is broken for '${marker:-<all>}': the passes claim $sum tests but the" \
         "marker selects $total. A test most likely declares a @pytest.mark.parallel(n) for an n the" \
         "workflow does not launch (it launches 1 $*), so nothing runs it. Add that n to the loop in" \
         "ci_pipeline.yml, or change the test."
    exit 1
fi
echo "  partition OK"
