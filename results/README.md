# Best Recorded Results

This directory keeps only the best run artifacts for each protocol on `bnci2014_001`.

## Within-subject (best)

- Run directory: `results/within_subject_tuned_da_step4/bnci2014_001_20260417_180456`
- Mean accuracy: `0.8237547893`
- Std accuracy: `0.1364329028`
- Mean kappa: `0.6454584012`
- Summary file: `results/within_subject_tuned_da_step4/bnci2014_001_20260417_180456/within_subject_results.json`

Reproduce:

```bash
./.venv/bin/python training/within_subject.py \
  --dataset bnci2014_001 \
  --epochs 50 \
  --lr 1e-3 \
  --min_lr 1e-4 \
  --weight_decay 1e-4 \
  --label_smoothing 0.0 \
  --augment_noise_std 0.0 \
  --augment_time_mask_ratio 0.0 \
  --early_stopping_patience 0 \
  --selection_metric accuracy \
  --use_da \
  --domain_loss_weight 1.65 \
  --da_lambda_gamma 10 \
  --loader_euclidean_align \
  --model_riemannian_reweight \
  --seed 42 \
  --deterministic \
  --output_dir results/within_subject_tuned_da_step4
```

## LOSO (best)

- Run directory: `results/loso_seed7_baseline/bnci2014_001_loso_20260418_055205`
- Mean accuracy: `0.7461419753`
- Std accuracy: `0.1119452252`
- Mean kappa: `0.4922839506`
- Summary file: `results/loso_seed7_baseline/bnci2014_001_loso_20260418_055205/loso_results.json`

Reproduce:

```bash
./.venv/bin/python training/loso.py \
  --dataset bnci2014_001 \
  --epochs 50 \
  --lr 1e-3 \
  --batch_size 32 \
  --seed 7 \
  --deterministic \
  --output_dir results/loso_seed7_baseline
```
