# Project Architecture Analysis

## 1. Executive Overview

This repository is organized as a compact EEG transfer-learning training stack built around PyTorch.
It uses a hybrid model (CNN + transformer encoder + adversarial domain head) and supports three practical workflows:

1. Local-array workflow via `main.py` (`.npy` files for `x`, `y`, `subject_id`)
2. Cross-subject evaluation (LOSO) via `training/loso.py`
3. Within-subject evaluation via `training/within_subject.py`

The architecture is modular and clear for experimentation, with code split into `data/`, `models/`, and `training/` packages.

---

## 2. High-Level Architecture

```text
Input Data
  |- Local arrays: data/x.npy, data/y.npy, data/subject_id.npy
  |- MOABB datasets: BNCI2014_001, PhysionetMI, Cho2017, Lee2019_MI
       |
       v
Data Layer (data/loader.py)
  |- EEGDataset
  |- split_eeg_data / create_dataloaders
  |- create_within_subject_dataloaders
  |- load_moabb_motor_imagery_dataset
       |
       v
Model Layer (models/)
  |- CNNBlock (temporal + depthwise spatial conv)
  |- EEGTokenizer (feature-to-token projection)
  |- ViTEncoder (CLS token + transformer blocks)
  |- TaskHead (class logits)
  |- GRL + DomainHead (subject/domain logits)
       |
       v
Training Layer (training/)
  |- pretrain.py   (task + adversarial domain loss)
  |- finetune.py   (partial unfreeze, task-only)
  |- loso.py       (held-out subject loops, metrics, logs)
  |- within_subject.py (per-subject train/test)
  |- utils.py      (alignment + lambda scheduler)
       |
       v
Artifacts
  |- checkpoints/*.pt
  |- results/*/*.json
  |- train.log
```

---

## 3. Package and Module Responsibilities

## `data/`

- `EEGDataset`: wraps `(x, y, subject_id)` into PyTorch tensors.
- `split_eeg_data`: supports either random stratified split or LOSO split.
- `create_dataloaders`: generic train/test dataloader creation.
- `create_within_subject_dataloaders`: filters one subject then stratified split.
- `load_moabb_motor_imagery_dataset`: MOABB ingestion with common channels and label mapping.

## `models/`

- `cnn.py`: EEG feature extractor with temporal conv, depthwise spatial conv, residual projection.
- `tokenizer.py`: converts CNN timewise features into transformer tokens.
- `vit.py`: lightweight transformer encoder with CLS token and optional positional embedding.
- `heads.py`: task classifier + gradient reversal layer + domain classifier.
- `model.py`: integrates preprocessing, feature extraction, and both heads into `EEGModel`.

## `training/`

- `pretrain.py`: adversarial pretraining (`task_loss + lambda * domain_loss`).
- `finetune.py`: loads pretrained weights, freezes most layers, updates last transformer block + task head.
- `loso.py`: complete LOSO experiment pipeline with logging, checkpointing, per-subject metrics.
- `within_subject.py`: per-subject supervised training/evaluation pipeline.
- `utils.py`: Euclidean alignment, Riemannian-inspired reweighting, DANN lambda scheduler.

## Top-level scripts

- `main.py`: simple local entrypoint for pretrain + optional finetune.
- `test.py`: dataset loading smoke-test using MOABB and common channels.

---

## 4. Runtime Flows

## A) `main.py` flow (local arrays)

1. Load config from `config.yaml` if present, else use dataclass defaults.
2. Read `x/y/subject_id` NumPy arrays.
3. Create train/validation loaders.
4. Build `EEGModel` from inferred dimensions.
5. Run adversarial pretraining.
6. Save checkpoint.
7. Evaluate validation accuracy.
8. Optionally run finetune and re-evaluate.

## B) `training/loso.py` flow

1. Load selected MOABB dataset.
2. For each held-out subject: split train/test with LOSO.
3. Train supervised classification model for that fold.
4. Compute accuracy and Cohen's kappa.
5. Save history + checkpoint + aggregate JSON summary.

## C) `training/within_subject.py` flow

1. Load MOABB dataset.
2. For each selected subject: split that subject's trials train/test.
3. Train and evaluate subject-specific model.
4. Save per-subject metrics and aggregate summary.

---

## 5. Model Architecture (Current)

Input shape assumption: `(B, C, T)`

1. **Signal alignment stage** (inside `EEGModel.forward`)
   - `euclidean_alignment`
   - `riemannian_reweight`

2. **CNN stage**
   - Temporal convolution over time axis.
   - Depthwise spatial convolution over channels.
   - Residual projection branch and ELU activation.
   - Average pooling and dropout.

3. **Tokenization stage**
   - Squeeze spatial axis and transpose to time-major token order.
   - Linear projection to embedding dimension.

4. **Transformer stage**
   - Prepend CLS token.
   - Add positional embeddings (if enabled).
   - Pass through `num_layers` transformer blocks.
   - Use CLS vector as global representation.

5. **Output heads**
   - `TaskHead`: motor imagery class logits.
   - `DomainHead` via `GRL`: subject/domain logits for adversarial invariance.

---

## 6. Experiment and Artifact Architecture

- Checkpoints are stored as `.pt` state dicts.
- Structured run artifacts are saved as JSON (`config`, per-subject history, final summary).
- LOSO and within-subject scripts each create timestamped run directories.
- Logging writes both console and file logs.

This setup is reproducibility-friendly for research iteration, though there is no central experiment registry layer yet.

---

## 7. Architectural Strengths

- Clear separation of data, model, and training concerns.
- Multiple evaluation protocols are already implemented (LOSO and within-subject).
- Model is modular and easy to swap components.
- Experiment scripts persist useful artifacts (logs, metrics, checkpoints).
- Dependency stack is aligned with EEG research tooling (MNE, MOABB, PyRiemann, PyTorch).

---

## 8. Architectural Gaps and Risks

- Two parallel entry styles (`main.py` vs protocol scripts) create duplicated training/evaluation logic.
- Configuration is fragmented (dataclass defaults + CLI args + optional YAML).
- `main.py` assumes local `.npy` files, while protocol scripts use MOABB; there is no unified dataset abstraction layer.
- `num_subjects = max(subject_id)+1` assumes near-contiguous IDs and can over-allocate domain classes.
- Alignment/reweighting inside model forward couples data preprocessing with architecture, making ablation and deployment harder.
- No formal tests/CI pipeline for regression safety.
- `README.md` is currently empty, so architecture and usage are discoverable only from source.

---

## 9. Current Architecture Verdict

The project currently follows a **research-oriented layered architecture**:

- **Data ingestion layer** (`data/`)
- **Neural architecture layer** (`models/`)
- **Protocol/training layer** (`training/`)
- **Script entry layer** (`main.py`, `training/*.py`, `test.py`)

It is well-structured for iterative experimentation and cross-subject EEG studies, but would benefit from unifying config + pipeline orchestration and formalizing documentation/testing for long-term maintainability.
