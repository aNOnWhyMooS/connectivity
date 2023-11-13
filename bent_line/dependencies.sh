#!/bin/sh
<<summary
Example usage:
  bash dependencies.sh 5 2 "sbatch line_with_bends.sbatch 0 2 bert_ft_qqp_6ep- 4000,4000 79,80 exp0 1 0"

The above command will launch 5 jobs. Each job is expected to use two dataloaders.
Each command will have the last field (the begin_from_dataloader) increase by 2, and the previous job
as a dependency.
summary

set -e

echo "Reading args.."
total_jobs=$1
dls_per_job=$2
cmd=$3
start_job=${4:-0}

echo "Starting.."
echo "Initial cmd: ${cmd}"
echo "Start job number: ${start_job}"
echo "Total jobs: ${total_jobs}"

for i in $(seq ${start_job} $((total_jobs + start_job - 1)));
do

    cmd=$(awk -F' ' -v i=$i -v d=${dls_per_job} '{ $NF = d*i; print }' <<< "${cmd}");

    if [ ! -z "${int_jid}" ];
    then
        cmd_with_dep=$(echo ${cmd} | sed "s/sbatch/sbatch --dependency=afterany:${int_jid}/");
    else
        cmd_with_dep=${cmd}
    fi;

    echo "Running: ${cmd_with_dep};"

    jid=$(eval "${cmd_with_dep}");
    int_jid=$(echo ${jid} | grep -Eo '[0-9]{8,}');

done;

