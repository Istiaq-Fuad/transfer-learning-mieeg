# EEG Motor Imagery Training (Simplified)

This repository focuses on one workflow: load MOABB motor imagery data, preprocess it, and run either LOSO or within-subject evaluation using the current EEG model.

## Quick-start

LOSO (leave-one-subject-out):

```bash
uv run training/run.py --protocol loso --dataset bnci2014_001 --subjects 1,2,3
```

Within-subject split:

```bash
uv run training/run.py --protocol within --dataset bnci2014_001 --subjects 1,2,3
```

Optional flags:

```bash
uv run training/run.py --protocol loso --dataset bnci2014_001 --data_path /path/to/mne_data \
 --epochs 70 --batch_size 32 --lr 5e-4 --weight_decay 1e-4 \
 --class_policy all --use_common_channels \
 --val_size 0.2 --patience 10 --min_delta 1e-4 --lr_schedule cosine \
 --within_cv_folds 5 --within_lr 3e-4 --within_label_smoothing 0.1
```

## Outputs

Each run writes results under the output directory (default: `results/`) with:

- `summary.json` with mean and per-subject metrics
- `history.json` with per-epoch training curves
- `train.log` with console logs

The entry point is [training/run.py](training/run.py).
