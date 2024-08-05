#!/bin/bash -x

if [ "$SHELL_SCRIPT" = "cupy" ] ; then {
    partition="develgpus"
} else {
    partition="devel"
}
fi

# General info on partition:
# sinfo -p $partition --noheader --format="%.15P %.5a %.10A"

part_up=$(sinfo -p $partition --noheader --format="%a")
nodes_avail=$(sinfo -p $partition --noheader --format="%A")

if [ "$part_up" = "up" ] && [ "$nodes_avail" != "0/0" ] ; then {
    echo "Partition up and nodes available"
    exit 0
} else {
    echo "Partition down or no nodes available"
    exit 100
}
fi
