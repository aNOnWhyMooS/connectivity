# Finetune Models

## Create Necessary Directories

```bash
mkdir sbatch_outs/
```

## Generate sbatch files

```bash
sbatch sbatch_maker.sbatch <user_name> <steps> <last_used_overlay_file_idx>
```

## Notes

``steps`` is the number of steps at which the pretrained model is to be taken, for finetuning further models.  
Currently done/ongoing(update here, if you start some new): [1600k, 1400k, 1200k, 1000k, 800k]  

When using first time, ``<last_used_overlay_file_idx>`` can be put as 0.  
