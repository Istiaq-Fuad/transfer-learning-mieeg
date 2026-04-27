# EEG Motor Imagery — Cross-Dataset Pretraining: Implementation Plan

**Goal:** Self-supervised pretraining on PhysioNet + Cho2017 + Lee2019, then zero-shot transfer to an unseen target dataset with a swappable classifier head. Beat the 74% LOSO baseline through pretrained representations that generalize across subjects and datasets without requiring aligned class labels.

---

## Pipeline at a Glance

```
PhysioNet + Cho2017 + Lee2019  (MOABB loaders)
  └─ Unified preprocessing  (250 Hz · 4–40 Hz · common channels · z-score)
       └─ MAE pretraining  (no labels, all subjects mixed)
            └─ Encoder backbone saved
                 └─ Attach new head  (N target classes)
                      └─ Gradual unfreeze fine-tuning
                           ├─ LOSO evaluation
                           ├─ Within-subject evaluation
                           └─ Reduced-data curve  (5 · 10 · 20 · 50 · 100 trials)
```

**Expected gains over 74% LOSO baseline:**

| Scenario | Expected gain |
|---|---|
| Pretrained encoder (frozen head) | +2–4% |
| Pretrained + gradual unfreeze | +4–8% |
| Reduced data (5–20 trials/class) | +10–15% |

---

## Phase 1 — Data Preparation

### 1.1 MOABB Integration

All three source datasets are loaded via the existing MOABB dataloader. The following modifications are required for this project:

**Datasets to load:**

| Dataset | MOABB class | Subjects | Classes |
|---|---|---|---|
| PhysioNet | `BNCI2014001` / `PhysionetMI` | 109 | 4 |
| Cho2017 | `Cho2017` | 52 | 2 |
| Lee2019 | `Lee2019_MI` | 54 | 2 |

**Required modifications to the existing MOABB loader:**

- Add a `return_labels=False` flag to suppress label loading during the pretraining phase. Labels must still be stored internally for fine-tuning but should not be passed to the pretraining dataloader.
- Add a `dataset_id` field to each returned sample so the model can log per-dataset reconstruction loss during pretraining (useful for monitoring, not for training).
- Add a `subject_id` field for downstream LOSO splitting.
- Ensure the loader returns raw epoched arrays of shape `(n_trials, C, T)` rather than pre-extracted features.

### 1.2 Unified Preprocessing Pipeline

Apply the following steps in order, identically across all three datasets. This uniformity is critical — cross-dataset variance in preprocessing will dominate the pretraining loss.

**Step 1 — Resampling.** Resample all signals to 250 Hz. Use `mne.io.Raw.resample(250)` or the MOABB-equivalent resampling argument.

**Step 2 — Bandpass filtering.** Apply a 4–40 Hz bandpass filter. This covers delta, theta, mu (8–12 Hz), and beta (13–30 Hz) bands. The mu and beta bands carry the primary motor imagery signal (ERD/ERS). Use a 4th-order Butterworth zero-phase filter.

**Step 3 — Common channel set.** Compute the intersection of electrode sets across all three datasets. Drop any channel not present in all three. The expected result is approximately 32–44 channels covering the standard 10-20 locations. Fix the channel ordering alphabetically or by standard 10-20 index to ensure positional embeddings remain consistent across datasets.

**Step 4 — Re-referencing.** Apply common average reference: subtract the mean across all channels at each time point.

**Step 5 — Epoching.** Extract epochs from 0 to 4 seconds post-cue onset. Apply a 500 ms pre-stimulus baseline correction (subtract the mean of the 500 ms window before cue onset). Final epoch shape per trial: `(C, 1000)` at 250 Hz.

**Step 6 — Artifact rejection.** Reject epochs with peak-to-peak amplitude exceeding 100 µV on any channel. Use MNE autoreject or a simple threshold pass.

**Step 7 — Normalization.** Apply per-channel, per-subject z-score normalization: subtract the channel mean and divide by the channel standard deviation. Compute statistics on the training set only — never on the test subject in LOSO. Store normalization parameters per subject for reproducibility.

**Step 8 — Label handling.** Store class labels in the sample metadata but do not pass them to the pretraining dataloader. Labels are loaded only at fine-tuning time, after the encoder weights are frozen and the new head is attached.

### 1.3 Output Format

Each subject's data should be accessible as:

```
X:       ndarray  (n_trials, C, T)     # e.g. (80, 44, 1000)
y:       ndarray  (n_trials,)          # stored, not used in pretraining
meta:
  dataset_id:   str    # "physionet" | "cho2017" | "lee2019"
  subject_id:   str    # "S001", "S002", ...
  sfreq:        int    # 250
  ch_names:     list   # common channel set, fixed ordering
```

### 1.4 Pretraining DataLoader

The pretraining dataloader must concatenate all trials from all subjects across all three datasets into a single pool and shuffle uniformly at each epoch. **This cross-dataset, cross-subject mixing within each batch is the single most important implementation detail for learning dataset-invariant representations.**

- Return only `(X, dataset_id)` — no labels.
- Use `shuffle=True` with a fixed global seed for reproducibility.
- Recommended batch size: 256–512.

### 1.5 Target Dataset DataLoader

For fine-tuning and evaluation, a separate loader handles the unseen target dataset:

- Accepts a list of subject IDs to include (used to construct LOSO splits).
- Returns `(X, y)` pairs with the target dataset's own class labels.
- Applies the same preprocessing pipeline as above, with normalization statistics computed only on the fine-tuning subjects (never on the test subject).

---

## Phase 2 — Model Architecture

### 2.1 Overview

The model consists of three separable components: a patch embedder, a Transformer encoder (the pretrained backbone), and a lightweight decoder used only during pretraining. At fine-tuning time the decoder is discarded and a new classification head is attached.

```
Input: (B, C, T) = (batch, 44 channels, 1000 samples)

[Pretrain path]
  EEGPatchEmbed   → (B, 8 patches, 256)
  Random masking  → keep 25% of patches
  MAE Encoder     → encoded visible patches
  MAE Decoder     → reconstructed masked patches
  Loss            → MSE on FFT magnitude of masked patches

[Fine-tune path]
  EEGPatchEmbed   → (B, 8 patches, 256)
  MAE Encoder     → full patch sequence (no masking)
  Mean pooling    → (B, 256)
  ClassifierHead  → (B, N_classes)
```

### 2.2 EEGPatchEmbed

Splits the temporal dimension into non-overlapping patches of 0.5 seconds (125 samples at 250 Hz), producing 8 patches per trial. Applies a linear projection across channels and a learned positional embedding across the patch dimension.

- Input: `(B, C, T)` = `(B, 44, 1000)`
- Channel projection: `Linear(C, embed_dim)` applied across time
- Patch split: unfold time axis with window=125, stride=125 → 8 patches
- Output: `(B, 8, 256)` + positional embedding

### 2.3 MAE Encoder

A standard Transformer encoder with random patch masking. During pretraining, 75% of patches are masked (replaced with a learnable mask token) and only the visible 25% are passed through the encoder.

| Hyperparameter | Value |
|---|---|
| Embedding dim | 256 |
| Encoder layers | 6–8 |
| Attention heads | 8 |
| FFN hidden dim | 1024 |
| Dropout | 0.1 |
| Mask ratio (pretrain) | 75% |
| Mask ratio (fine-tune) | 0% |

### 2.4 MAE Decoder

A lightweight 2-layer Transformer decoder used only during pretraining. It takes the encoded visible patches plus learnable mask tokens at the masked positions and reconstructs all 8 patches. The decoder is discarded entirely after pretraining.

- Loss: MSE on the FFT magnitude spectrum of the masked patches (log power spectrum preferred).
- Computing loss in the frequency domain forces the encoder to learn spectral structure (mu/beta band power) rather than raw waveform shape.

### 2.5 ClassifierHead

A minimal head that replaces the decoder after pretraining. It is instantiated fresh for each target dataset, which cleanly handles the class label mismatch across datasets — the encoder is label-agnostic, so only the head size needs to match the target.

```
Mean pooling over patch dimension  → (B, 256)
LayerNorm(256)
Dropout(0.3)
Linear(256, N_classes)
```

`N_classes` is set at instantiation time based on the target dataset (e.g., 2 for Cho2017-style binary MI, 4 for PhysioNet-style four-class MI).

---

## Phase 3 — Self-Supervised Pretraining

### 3.1 Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW, β1=0.9, β2=0.95, weight_decay=0.05 |
| Learning rate | 1e-4, cosine decay with 20-epoch linear warmup |
| Batch size | 256–512 |
| Epochs | 200–500 (early stop on val reconstruction loss) |
| Gradient clipping | max norm = 1.0 |
| Validation split | 10% of subjects per dataset, held out throughout pretraining |
| Checkpointing | Save encoder weights every 50 epochs |

### 3.2 Loss Function

Reconstruction loss is computed in the frequency domain on the masked patches only:

```
target = log(|FFT(masked_patches)| + ε)
pred   = decoder(encoder(visible_patches), mask_positions)
loss   = MSE(pred, target)
```

Computing loss on the log power spectrum rather than raw signal amplitude has two advantages: it down-weights high-amplitude low-frequency drift that dominates the MSE if computed on raw samples, and it forces the model to learn the spectral structure of mu and beta band oscillations directly.

### 3.3 Monitoring

Log the following per epoch to verify healthy pretraining:

- Overall reconstruction loss (train and val)
- Per-dataset reconstruction loss (physionet / cho2017 / lee2019 separately) — all three should decrease at similar rates; a large gap indicates preprocessing inconsistency
- Gradient norm — should stay below 1.0
- Val loss / train loss ratio — should stay below 1.3; higher indicates overfitting to source subjects

Visually inspect reconstructed patches from a held-out subject every 50 epochs. The reconstruction should capture the bandpower envelope shape but not overfit to single-trial noise.

### 3.4 Recommended Ablations

Run the following ablations after the first successful pretraining run, before proceeding to fine-tuning:

- **Mask ratio:** 50% vs 75% — 75% is expected to produce better representations by forcing the encoder to learn more robust contextual features.
- **Loss domain:** time-domain MSE vs frequency-domain MSE — frequency-domain is expected to perform better for MI.
- **Patch size:** 0.5 s (125 samples) vs 1.0 s (250 samples) — 0.5 s captures faster transients in the mu/beta range.
- **Dataset pooling:** pretrain on all 3 vs each dataset individually — pooling should outperform any single dataset on the cross-dataset evaluation.

---

## Phase 4 — Head Swapping and Fine-Tuning

### 4.1 Gradual Unfreeze Strategy

After pretraining, discard the decoder, load the saved encoder weights, and attach a fresh `ClassifierHead` with `n_classes` set to the target dataset's number of classes. Fine-tune using the following three-step schedule:

| Step | Trainable parameters | Learning rate | Epochs | Purpose |
|---|---|---|---|---|
| 1 — Head only | New classifier head | 1e-3 | 5–10 | Calibrate head without corrupting encoder |
| 2 — Last 2 blocks | Head + last 2 Transformer blocks | 5e-5 | 10–20 | Adapt upper layers to target distribution |
| 3 — Full model | All layers | 1e-5 | 10–20 | Optional — use only for within-subject, skip for LOSO |

Use early stopping with patience=10 on the validation accuracy at each step. For LOSO, use steps 1 and 2 only — step 3 risks overfitting when fine-tuning data is limited.

### 4.2 Regularization

- Dropout 0.3 on the classifier head.
- Label smoothing 0.1.
- Light data augmentation during fine-tuning: Gaussian noise (σ=0.01) and random time shift (±20 samples). Do not use channel dropout during fine-tuning — the channel set is already fixed.

### 4.3 Class Label Mismatch

Because the encoder is trained with no labels, there is no mismatch problem at the encoder level. The only action required at fine-tuning time is instantiating the classifier head with the correct `n_classes` for the target dataset. Never reuse or transfer head weights across datasets.

---

## Phase 5 — Evaluation Protocols

### 5.1 LOSO — Leave-One-Subject-Out

This is the primary benchmark. The procedure mirrors the existing 74% baseline evaluation exactly, with the pretrained encoder replacing the from-scratch encoder.

```
for test_subject in target_dataset.subjects:

    train_subjects = all subjects except test_subject

    model = pretrained_encoder + fresh_head(N_classes)
    fine_tune(model, train_subjects, strategy=steps_1_and_2_only)

    acc[test_subject] = evaluate(model, test_subject)

report: mean(acc) ± std(acc)
```

Report mean ± std accuracy across all test subjects. Compare against three baselines: from-scratch LOSO (existing 74%), pretrained-frozen-head LOSO, and pretrained-gradual-unfreeze LOSO.

### 5.2 Within-Subject

Provides the upper bound on performance for each subject. Use 5-fold cross-validation within each subject (not a single train/test split).

```
for subject in target_dataset.subjects:
    for fold in 5-fold CV on subject.trials:

        model = pretrained_encoder + fresh_head(N_classes)
        fine_tune(model, train_fold, strategy=all_3_steps)
        acc[subject][fold] = evaluate(model, test_fold)

report: mean(acc) ± std(acc) across subjects
```

Expected range: 85–90% for two-class MI.

### 5.3 Reduced Data Learning Curve

This experiment provides the clearest evidence of pretraining value and is important for the publication. It shows how quickly the pretrained model adapts compared to a from-scratch model as labelled data is reduced.

```
N_values = [5, 10, 20, 50, 100]   # labelled trials per class

for N in N_values:
    for repeat in range(10):       # 10 random seeds for stable estimates

        subset = sample_balanced(train_data, N_per_class=N, seed=repeat)

        # Pretrained model
        model_pt = pretrained_encoder + fresh_head(N_classes)
        fine_tune(model_pt, subset)
        acc_pretrained[N][repeat] = evaluate(model_pt, test_data)

        # From-scratch baseline
        model_sc = EEGClassifier(N_classes)   # same architecture, random init
        train_from_scratch(model_sc, subset)
        acc_scratch[N][repeat] = evaluate(model_sc, test_data)

report: mean ± std for each N, for both pretrained and scratch
plot:   accuracy vs N (two curves, with std shading)
```

The gap between the pretrained and scratch curves at low N (5–20 trials) is the key result for the paper. Expect +10–15% at N=5.

### 5.4 Expected Results Summary

| Model | LOSO | Within-subject | @ 10 trials/class |
|---|---|---|---|
| From scratch (current baseline) | 74% | ~85% | ~55% |
| Pretrained — frozen head | ~76–78% | ~86% | ~65% |
| Pretrained — gradual unfreeze | **~78–82%** | ~88% | **~70–75%** |

---

## Hyperparameter Reference

### Preprocessing

| Parameter | Value |
|---|---|
| Sampling rate | 250 Hz |
| Bandpass | 4–40 Hz, 4th-order Butterworth |
| Epoch window | 0–4 s post-cue |
| Baseline correction | −500–0 ms pre-cue |
| Re-reference | Common average |
| Artifact threshold | 100 µV peak-to-peak |
| Normalization | Per-channel, per-subject z-score |
| Common channels | Intersection across all 3 datasets (~32–44) |

### Model

| Parameter | Value |
|---|---|
| Patch size | 125 samples (0.5 s at 250 Hz) |
| Number of patches | 8 per trial |
| Embedding dimension | 256 |
| Encoder layers | 6–8 |
| Attention heads | 8 |
| FFN hidden dim | 1024 |
| Dropout (encoder) | 0.1 |
| Mask ratio | 75% (pretrain) / 0% (fine-tune) |
| Decoder layers | 2 (pretrain only) |

### Pretraining

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| β1, β2 | 0.9, 0.95 |
| Weight decay | 0.05 |
| Learning rate | 1e-4, cosine decay |
| Warmup | 20 epochs |
| Batch size | 256–512 |
| Max epochs | 200–500 |
| Gradient clip | max norm = 1.0 |

### Fine-tuning

| Parameter | Value |
|---|---|
| Step 1 LR | 1e-3 |
| Step 2 LR | 5e-5 |
| Step 3 LR | 1e-5 (within-subject only) |
| Batch size | 32–64 |
| Dropout (head) | 0.3 |
| Label smoothing | 0.1 |
| Early stop patience | 10 epochs |