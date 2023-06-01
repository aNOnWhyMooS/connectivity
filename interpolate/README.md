## Making sbatch files

Make sure to mention the model numbers which may not have been completely trained in the ``half_saved_models`` argument in ``sbatch_maker.sbatch``, [here](https://github.com/aNOnWhyMooS/connectivity/blob/f7bb184ad85d990040e82d42bdeeef89d5773c19/interpolate/sbatch_maker.sbatch#L12) before running the commands below. 

A sample command:

```bash
cd $SCRATCH
cd connectivity/interpolate
mkdir sbatch_outs

sbatch sbatch_maker.sbatch $USER mnli test 10000 0 NLI/roberta_large Jeevesh8/corr_init_shuff_clipped_warmed_wd0_pnt_01_seq_len_128_roberta-large_mnli_ft_ flax roberta-large
```

The above command produces files for running interpolations between models at 10000 training steps, with prefix as ``Jeevesh8/corr_init...``. The type of models to load from hub also needs to be specified, here,``flax``. The argument ``NLI/roberta_large`` is the path within ``../../constellations/logs/`` directory where the interpolation logs are to be saved. The last argument is the base model from which all these models were finetuned. The ``0`` after 10000 is a dummy argument, for now.

Next, we move all the generated sbatch files to a directory, and run them.
```bash
mkdir -p sbatch_files/roberta_large
mkdir sbatch_files/roberta_large/10000steps
mv mnli_test_corr_init* sbatch_files/roberta_large/10000steps
for i in {1..20}; do sbatch sbatch_files/roberta_large/10000steps/mnli_test_corr_init_shuff_clipped_warmed_wd0_pnt_01_seq_len_128_roberta-large_mnli_ft__ft-${i}.sbatch; done;
```
