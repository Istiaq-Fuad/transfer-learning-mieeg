# Project Architecture Analysis

## 1. Executive Overview

This repository is a research-focused EEG motor-imagery training stack built on PyTorch.
The current codebase centers on one shared model architecture and three runnable experiment styles:

1. Local `.npy` pretrain + optional finetune via `main.py`
2. Cross-subject LOSO protocol via `training/loso.py`
3. Within-subject protocol via `training/within_subject.py`

Relative to the previous snapshot, the architecture now includes stronger reproducibility metadata utilities and richer dataloader options (alignment hooks, deterministic generators, optional subject-balanced sampling), while still keeping a compact package split across `data/`, `models/`, `training/`, and `utils/`.

---

## 2. High-Level Architecture

```text
Input Sources
  |- Local arrays: data/x.npy, data/y.npy, data/subject_id.npy
  |- MOABB datasets: BNCI2014_001, PhysionetMI, Cho2017, Lee2019_MI
       |
       v
Data Layer (data/loader.py)
  |- EEGDataset tensor wrapper
  |- split_eeg_data (random split or LOSO)
  |- create_dataloaders (alignment + optional subject-balanced sampling)
  |- create_within_subject_dataloaders
  |- load_moabb_motor_imagery_dataset
  |- subsample_train_trials_per_subject_class
       |
       v
Model Layer (models/)
  |- CNNBlock (temporal conv + depthwise spatial conv + residual)
  |- EEGTokenizer (feature-to-token projection)
  |- ViTEncoder (CLS token + transformer stack)
  |- TaskHead
  |- GRL + DomainHead
  |- EEGModel (wires full forward graph)
       |
       v
Training / Protocol Layer (training/)
  |- pretrain.py (adversarial pretraining)
  |- finetune.py (selective unfreezing, task-only update)
  |- loso.py (cross-subject evaluation protocol)
  |- within_subject.py (per-subject protocol)
  |- utils.py (alignment/reweighting/schedule math)
       |
       v
Reproducibility + Artifacts
  |- utils/reproducibility.py (seed control + metadata hash/id)
  |- checkpoints/*.pt
  |- results/**/{config,metadata,*_history,*_results}.json
  |- train.log
```

---

## 3. Package and Module Responsibilities

## `data/`

- `EEGDataset`: validates length consistency and returns `(x, y, subject_id)` tensors.
- `split_eeg_data`: supports stratified random split or LOSO split by held-out subject.
- `create_dataloaders`:
  - Optional train-statistics Euclidean alignment applied to both train/test.
  - Optional `WeightedRandomSampler` for subject-balanced training batches.
  - Deterministic `torch.Generator` support when seed and deterministic mode are enabled.
- `create_within_subject_dataloaders`: subject filter + stratified split + same optional alignment/sampling controls.
- `load_moabb_motor_imagery_dataset`: dataset dispatch, common-channel filtering, label remapping to integer classes, and subject selection.
- `subsample_train_trials_per_subject_class`: class-stratified per-subject trial reduction utility for reduced-data experiments.

## `models/`

- `cnn.py` (`CNNBlock`): temporal convolution, depthwise spatial convolution, residual projection, ELU, pooling, dropout.
- `tokenizer.py` (`EEGTokenizer`): reshapes CNN features into token sequence and projects to embedding dim.
- `vit.py` (`ViTEncoder`): pre-norm transformer blocks with CLS token and optional learned positional embedding.
- `heads.py`: `TaskHead`, `DomainHead`, and autograd-based `GRL`.
- `model.py` (`EEGModel`): applies alignment/reweighting in forward, then CNN -> tokenizer -> ViT -> task and domain heads.

## `training/`

- `pretrain.py`: adversarial pretraining with DANN-style schedule (`task + lambda * domain`).
- `finetune.py`: loads pretrained weights, freezes most layers, updates last transformer block + task head.
- `loso.py`: full MOABB LOSO pipeline with per-fold history, checkpoints, kappa/accuracy metrics, and run-level summary JSON.
- `within_subject.py`: per-subject train/test workflow with same metric and artifact pattern.
- `utils.py`: linear algebra utilities for Euclidean alignment, covariance-based reweighting, and lambda scheduler.

## `utils/`

- `reproducibility.py`: consolidated seed setup, data-loader generator creation, config hashing, and experiment metadata persistence.

## Top-level scripts

- `main.py`: local-array training entrypoint with YAML-or-default config, checkpointing, validation evaluation, and optional finetune stage.
- `test.py`: manual MOABB loading script with hardcoded environment paths for data access smoke checks.

---

## 4. Runtime Flows

## A) `main.py` (local arrays + adversarial pretrain)

1. Load config from `config.yaml` when present, else dataclass defaults.
2. Set global seeds and deterministic flags via reproducibility utility.
3. Load `x`, `y`, `subject_id` arrays and create train/validation dataloaders.
4. Build `EEGModel` from inferred channel/class/subject counts.
5. Run adversarial pretraining loop.
6. Save checkpoint and write `metadata.json` with config hash + experiment id.
7. Evaluate validation accuracy.
8. Optionally load checkpoint and run finetuning pass.

## B) `training/loso.py` (MOABB LOSO protocol)

1. Parse CLI args and configure reproducibility.
2. Load MOABB dataset and selected subjects.
3. For each held-out subject:
   - Build LOSO train/test loaders.
   - Instantiate a fresh model.
   - Train with supervised task loss.
   - Evaluate accuracy and Cohen's kappa per epoch.
   - Save subject checkpoint + history JSON.
4. Save aggregate `loso_results.json` plus `config.json` and `metadata.json`.

## C) `training/within_subject.py` (subject-specific protocol)

1. Parse CLI args and load selected MOABB dataset.
2. For each subject:
   - Build within-subject train/test loaders.
   - Train and evaluate subject-specific model.
   - Save checkpoint + per-subject history JSON.
3. Save aggregate `within_subject_results.json` plus run config/metadata.

---

## 5. Model Architecture (Current)

Input expectation: `(B, C, T)`.

1. Preprocessing in `EEGModel.forward`:
   - `euclidean_alignment(x)`
   - `riemannian_reweight(x)`
2. CNN feature extraction:
   - Temporal conv on time axis.
   - Depthwise spatial conv across channels.
   - Residual projection with temporal-size alignment guard.
   - ELU + average pooling + dropout.
3. Tokenization:
   - Convert `(B, F, 1, T')` to `(B, T', D)`.
4. Transformer encoding:
   - Prepend CLS token.
   - Add learned positional embedding.
   - Pass through `num_layers` transformer blocks.
   - Use normalized CLS vector.
5. Dual heads:
   - Task classification head.
   - Domain classification head fed through GRL for adversarial invariance.

---

## 6. Experiment and Artifact Architecture

- Each protocol run creates a timestamped run directory.
- Artifacts are JSON-first: run config, metadata, per-subject epoch history, and aggregate metrics.
- Model checkpoints are saved as PyTorch `state_dict` files.
- Protocol scripts emit both terminal logs and persistent `train.log` files.
- Metadata includes stable config hash and experiment id, improving reproducibility traceability.

---

## 7. Architectural Strengths

- Clean modular layering (`data`, `models`, `training`, `utils`).
- Shared core model reused by all protocols.
- Stronger reproducibility support than earlier version (seed orchestration + config hashing + metadata files).
- Protocol scripts are self-contained and generate rich run artifacts for later analysis.
- Data layer now supports both deterministic loading behavior and optional subject-balanced sampling.
- Subsampling utility exists for reduced-trial studies without touching model code.

---

## 8. Architectural Gaps and Risks

- Pipeline duplication remains: `main.py`, LOSO, and within-subject each maintain separate train/eval orchestration.
- Potential preprocessing duplication: Euclidean alignment can be applied in dataloaders and again inside `EEGModel.forward`.
- Domain class sizing still relies on `max(subject_id) + 1`, which assumes contiguous IDs.
- `main.py` is local-array-centric, while protocol scripts are MOABB-centric; there is still no single dataset abstraction.
- Reduced-transfer experiment outputs exist under `results/reduced_transfer_ablation`, but no corresponding runnable pipeline script is present in `training/`.
- `test.py` is environment-specific (hardcoded filesystem paths), limiting portability as a validation entrypoint.
- `README.md` is empty, so usage and architecture discovery remain source-driven.
- There is no formal automated test suite or CI gate to detect regressions.

---

## 9. Current Architecture Verdict

The project currently follows a pragmatic research architecture with:

- Data ingestion and split utilities in `data/`
- Reusable model components in `models/`
- Protocol-specific runners in `training/`
- Reproducibility helpers in `utils/reproducibility.py`

It is effective for iterative EEG experimentation and result logging. The next maintainability step is to unify experiment orchestration (single configurable runner), separate preprocessing concerns cleanly from model forward, and add lightweight automated tests plus baseline documentation.
