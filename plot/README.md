# Plotting The Graphs

## Heatmaps

Example:

```bash
python3 interpol_heatmap.py --order_by cluster --perf_metric lexical_overlap_onlyNonEntailing\
                    --interpol_log_dir ../logs/NLI/feather_berts/retrained_interpolate/\
                    --eval_mods_prefix ../logs/NLI/feather_berts/retrained_eval/hans_eval@36813steps/bert-base-uncased_mnli_ft_\
                    --eval_mods_suffixes 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31\
                    32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66\
                    67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99\
                    --emb_acc_corr
```

## Plotting Score Distributions of Models

Example:

```bash
python3 plot_score_dists.py --grp1_name original --eval_dir_mnli1 ../logs/NLI/feather_berts/mnli_eval\
                            --eval_dir_hans1 ../logs/NLI/feather_berts/hans_eval --grp2_name retrained\
                            --eval_dir_mnli2 ../logs/NLI/feather_berts/retrained_eval/mnli_eval@36813steps\
                            --eval_dir_hans2 ../logs/NLI/feather_berts/retrained_eval/hans_eval@36813steps\
                            --save_file_prefix original_vs_retrained_fbs
```

## Plotting Peak-Valley-Plain interpolations

Example:

```bash
python3 peak_valley_plains.py --perf_metric lexical_overlap_onlyNonEntailing\
                              --interpol_log_dir ../logs/NLI/feather_berts/interpolations/mnli_validation_matched/\
                              --eval_mods_prefix ../logs/NLI/feather_berts/hans_eval/bert_ --eval_mods_suffixes\
                              00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27\
                              28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55\
                              56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83\
                              84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99
```
