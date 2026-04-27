# Best Recorded Results

This directory keeps only the best run artifacts for `bnci2014_001`.

## Within-subject (best, DA)

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

## LOSO (best, without DA)

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

## LOSO (best, with DA)

- Run directory: `results/loso_arch_step0_base/bnci2014_001_loso_20260418_194906`
- Mean accuracy: `0.7457561728`
- Std accuracy: `0.1029738518`
- Mean kappa: `0.4915123457`
- Summary file: `results/loso_arch_step0_base/bnci2014_001_loso_20260418_194906/loso_results.json`

Reproduce:

```bash
./.venv/bin/python training/loso.py \
  --dataset bnci2014_001 \
  --epochs 50 \
  --lr 1e-3 \
  --batch_size 32 \
  --use_da \
  --domain_loss_weight 0.3 \
  --da_lambda_gamma 4 \
  --seed 42 \
  --deterministic \
  --output_dir results/loso_arch_step0_base
```

## Retuning Plan For New Architecture

Use this plan after enabling new architecture switches like `--temporal_kernels` or `--use_attention_pool`.

1. Stabilize feature preprocessing
   - Keep loader-level alignment on: `--loader_euclidean_align`
   - Keep in-model batch-wise transforms off for architecture experiments:
     `--model_pre_align_only`

2. Retune DA pressure before architecture knobs
   - For each candidate architecture, sweep:
     - `--domain_loss_weight`: `0.10 0.20 0.30`
     - `--da_lambda_gamma`: `2 3 4`
   - Start with `--epochs 50`, then extend best candidates to `--epochs 70`.

3. Multi-scale stem retuning
   - Try:
     - `--temporal_kernels 16 32 64 128 --multiscale_preserve_capacity`
     - `--temporal_kernels 32 64 128 --multiscale_preserve_capacity`
   - If unstable, reduce LR to `5e-4`.

4. Attention pooling retuning
   - Use:
     - `--use_attention_pool --learnable_attention_mix --attention_mix_init 0.7`
   - Compare against fixed mixes:
     - `--use_attention_pool --attention_mix_init 0.8`
     - `--use_attention_pool --attention_mix_init 0.6`

5. Strong domain head retuning (only after 1-4)
   - Start conservative:
     - `--domain_head_hidden_dim 64 --domain_head_layers 2 --domain_head_dropout 0.1`
   - Add CNN-level head only if above improves:
     - `--use_cnn_domain_head --cnn_domain_weight 0.2`

6. Robust selection rule (required)
   - Evaluate each finalist on seeds `7, 13, 42`.
   - Pick winner by:
     1) highest mean LOSO accuracy,
     2) lower std accuracy,
     3) higher mean kappa.

## Supervised Pretraining Transfer Protocol

Use this protocol before making reduced-data claims. The LOSO trainer now selects
epochs on a source-subject validation split and reports the old best-test number
separately as `legacy_best_test_*`.

Pretrain:

```bash
uv run python training/pretrain_cross_dataset.py \
  --source_datasets physionetmi cho2017 lee2019_mi \
  --pretrain_mode supervised \
  --domain_mode subject \
  --validation_strategy subject_fold \
  --subject_val_folds 5 \
  --subject_val_fold_index 0 \
  --loader_euclidean_align \
  --model_pre_align_only \
  --subject_balanced_sampling \
  --epochs 80 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --seed 42 \
  --deterministic
```

Full-data LOSO transfer:

```bash
uv run python training/loso.py \
  --dataset bnci2014_001 \
  --init_checkpoint results/pretrain_cross_dataset/<run>/checkpoints/pretrain_best.pt \
  --epochs 50 \
  --lr 1e-3 \
  --source_val_size 0.2 \
  --selection_metric accuracy \
  --loader_euclidean_align \
  --model_pre_align_only \
  --seed 42 \
  --deterministic
```

Reduced-data comparison after full-data recovery:

```bash
for fraction in 0.5 0.25 0.1; do
  uv run python training/loso.py \
    --dataset bnci2014_001 \
    --init_checkpoint results/pretrain_cross_dataset/<run>/checkpoints/pretrain_best.pt \
    --train_fraction "$fraction" \
    --epochs 50 \
    --lr 1e-3 \
    --source_val_size 0.2 \
    --loader_euclidean_align \
    --model_pre_align_only \
    --seed 42 \
    --deterministic
done
```
