#/bin/bash
<<com
Sample command:
  bash clean_and_rerun.sh sbatch_files/long_qqp/2000steps/qqp_validation_bert_ft_qqp_6ep-_ft-\
        ../../constellations/logs/NLI/long_qqp_berts/qqp_interpol@2000steps qqp validation

Detects and re-runs failed (which didn't produce a .pkl file) interpolation jobs using the logs
directory and the directory with sbatch files.
com

sbatch_files_prefix=$1
log_directory=$2
dataset=$3
split=$4

# Remove logs of failed jobs
for f in ${log_directory}/*interpolate*;
do
	if [[ $(tail -n 1 $f) != "Wrote the values"* && ${f: -4} != ".err" ]];
	then
		echo "Incomplete logs at: $f";
		rm $f;
		rm ${f:0:-4}.err;
	fi;
done;

# Restart failed jobs
for i in `seq 0 19`;
do
	if [[ -f ${log_directory}/interpol_${dataset}_${split}${i}.pkl ]];
	then
		echo "Job $((i+1)) successful!";
	else
		echo "Job $((i+1)) unsuccessful. Restarting.."
		sbatch ${sbatch_files_prefix}$((i+1)).sbatch;
	fi;
done;
