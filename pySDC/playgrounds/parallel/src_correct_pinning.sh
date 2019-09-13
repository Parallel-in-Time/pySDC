#!/usr/bin/env bash
# Script for a correct pinning w/ regard to hardware threads shared by a process
# using SLURM variables. Also needs an additional HWT multiplicity set in the
# environment.
# To share two hardware threads of each core for a process:
#    ...
#    export HWT=2
#    export PIN=`correct_pinning.sh`
#    ...
#    srun $PIN ...
#    ...

# echo settings? This will break srun integration...
ECHO=${VERBOSE:-false}
# to fix the integration, try
#    export PIN=`correct_pinning.sh | grep cpu_bind`
#

# run for JURECA cluster or booster?
# JUWELS should be covered anyway.
MCA="CLS"
echo "$SLURM_JOB_PARTITION" | grep -q booster && MCA="BOO"

function print_config() {
`$ECHO` && echo -e "\nHARDWARE CONFIG:"
`$ECHO` && echo "cores per node: $PHYS_CORES_NODE"
`$ECHO` && echo "CPUs per node: $SOCKETS"
`$ECHO` && echo "cores per CPU: $PHYS_CORES_CPU"
`$ECHO` && echo "hardware threads per core: $SMT"
`$ECHO` && echo "hardware threads per node: $SLURM_CPUS_ON_NODE"
`$ECHO` && echo -e "\nJOB CONFIG:"
`$ECHO` && echo "tasks per node: $SLURM_NTASKS_PER_NODE"
`$ECHO` && echo "hardware threads per task: $SLURM_CPUS_PER_TASK"
`$ECHO` && echo "shared hardware threads per process: $HWT"
}

function pin_cluster() {
SOCKETS=2
SMT=2
PHYS_CORES_NODE=$(($SLURM_CPUS_ON_NODE/$SMT))
PHYS_CORES_CPU=$(($PHYS_CORES_NODE/$SOCKETS))

print_config

# exit straight away if we can't evenly distribute threads
if [ $(($(($SLURM_CPUS_PER_TASK/$HWT))*$HWT)) != $SLURM_CPUS_PER_TASK ]
then
   `$ECHO` && echo "No nice disitribution of threads possible"
   exit 1
fi

CPUid=0
MASK="--cpu_bind=mask_cpu:"
# loop per process on each node
for PROC in `seq 1 $SLURM_NTASKS_PER_NODE`
do
   MAP=""
   `$ECHO` && echo "process $PROC"
   for CORE in `seq 1 $(($SLURM_CPUS_PER_TASK/$HWT))`
   do
      CPUid_=$CPUid
      for HW in `seq 1 $HWT`
      do
	 MAP="$MAP,$CPUid_"
	 ((CPUid_+=$PHYS_CORES_NODE))
      done
      ((CPUid++))
   done
   MAP_=`echo $MAP | sed  's/,/2^/' | sed 's/,/+2^/g'`
   MAP=`echo $MAP | sed  's/,//'`
   `$ECHO` && printf "map for process $PROC: %s\n" $MAP
   MASK="$MASK,0x"`echo "obase=16; $MAP_" | bc`
done
MASK=`echo $MASK | sed 's/:,/:/'`
echo $MASK
}

function pin_booster() {
SOCKETS=1
SMT=4
PHYS_CORES_NODE=$(($SLURM_CPUS_ON_NODE/$SMT))
PHYS_CORES_CPU=$(($PHYS_CORES_NODE/$SOCKETS))

print_config

# exit straight away if we can't evenly distribute threads
if [ $(($(($SLURM_CPUS_PER_TASK/$HWT))*$HWT)) != $SLURM_CPUS_PER_TASK ]
then
   `$ECHO` && echo "No nice disitribution of threads possible"
   exit 1
fi

CPUid=0
MASK="--cpu_bind=mask_cpu:"
# loop per process on each node
for PROC in `seq 1 $SLURM_NTASKS_PER_NODE`
do
   MAP=""
   `$ECHO` && echo "process $PROC"
   for CORE in `seq 1 $(($SLURM_CPUS_PER_TASK/$HWT))`
   do
      CPUid_=$CPUid
      for HW in `seq 1 $HWT`
      do
	 MAP="$MAP,$CPUid_"
	 ((CPUid_+=$PHYS_CORES_CPU))
      done
      ((CPUid++))
      if [ $CPUid -eq $PHYS_CORES_CPU ] && [ $HWT -eq 2 ]
      then
	 ((CPUid+=$PHYS_CORES_CPU))
      fi
   done
   MAP_=`echo $MAP | sed  's/,/2^/' | sed 's/,/+2^/g'`
   MAP=`echo $MAP | sed  's/,//'`
   `$ECHO` && printf "map for process $PROC: %s\n" $MAP
   MASK="$MASK,0x"`echo "obase=16; $MAP_" | bc`
done
MASK=`echo $MASK | sed 's/:,/:/'`
echo $MASK
}

if [ $MCA == "CLS" ] 
then
   pin_cluster
elif [ $MCA == "BOO" ]
then
   pin_booster
fi

exit 0
