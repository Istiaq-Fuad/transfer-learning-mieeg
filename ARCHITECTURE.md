# Architecture and Data Flow

This repository implements a simplified EEG motor imagery training pipeline. It is designed to load MOABB motor imagery datasets, preprocess them, and run either leave-one-subject-out (LOSO) or within-subject training/evaluation using the current EEG model.

> Update this file whenever the codebase changes.

## 1. Entry point

The main entry point is `training/run.py`.

- `python training/run.py --protocol loso --dataset bnci2014_001`
- `python training/run.py --protocol within --dataset bnci2014_001`

The command-line interface controls:

- `--protocol`: `loso` or `within`
- `--dataset`: MOABB dataset name
- `--subjects`: optional comma-separated list of subject IDs
- `--class_policy`: `all` or `left_right`
- `--use_common_channels`: use `_COMMON_CHANNELS` for consistent channel subsets
- `--data_path`: optional path for MOABB/MNE data caches
- `--val_size`: validation split size inside the training set
- `--within_cv_folds`: number of folds for within-subject CV (1 disables CV)
- Hyperparameters: `--epochs`, `--batch_size`, `--lr`, `--weight_decay`, etc.

## 2. Data flow

### 2.1 MOABB dataset loading

`training/run.py` calls `data.loader.load_moabb_motor_imagery_dataset(...)`.

- Supported datasets include `bnci2014_001`, `physionetmi`, `cho2017`, and `lee2019_mi`.
- Data is loaded using MOABB and MNE.
- Standard preprocessing is applied via the loader:
  - resampling to 250 Hz
  - bandpass filtering between 4 Hz and 40 Hz
  - by default, all EEG channels available in each dataset (no EOG channels)
  - optional common channel set via `_COMMON_CHANNELS` if enabled in options
  - returns epoched data in shape `(N, C, T)`

The loader returns:

- `x`: EEG data as `numpy.ndarray` shaped `(trials, channels, time)`
- `y`: integer class labels shaped `(trials,)` mapped from all available MI classes
- `subject_id`: subject IDs shaped `(trials,)`
- `available_subjects`: loaded subject list

### 2.2 Dataset splitting

The training protocol chooses one of two split functions in `data/loader.py`:

- `create_dataloaders(...)` for LOSO
  - if `loso_subject` is provided, all trials from that subject become the test set
  - other subjects are used for training
- `create_within_subject_dataloaders(...)` for within-subject evaluation
  - data for the target subject is split into train/test folds
  - an additional validation split is carved out of the training set

Both splitters use:

- stratified splits on class labels
- optional Euclidean alignment via `training.utils.fit_euclidean_alignment(...)`
- optional subject-balanced sampling in the training loader

### 2.3 Torch dataset abstraction

`data/loader.py` defines `EEGDataset`.

- It stores `x`, `y`, and `subject_id` as PyTorch tensors
- `__getitem__` returns `(x, y, subject_id)`

The training loop consumes batches of `(x, y, _)`.

## 3. Model architecture

The main model is `models.model.EEGModel`.

### 3.1 Input

- expects input tensor shape `(B, C, T)`
- performs optional Riemannian-inspired reweighting:
  - `training.utils.riemannian_reweight(...)`

### 3.2 CNN feature extractor

- `models.cnn.CNNBlock`
- temporal convolution branches with configurable kernel sizes
- depthwise spatial convolution across channels
- channel attention block
- residual path + temporal pooling
- outputs shape `(B, F, 1, T')`

### 3.3 Tokenization

- `models.tokenizer.EEGTokenizer`
- squeezes the spatial dimension and projects `(B, T', F)` to `(B, T', D)`
- produces transformer input tokens

### 3.4 Transformer encoder

- `models.vit.ViTEncoder`
- prepends a learnable CLS token
- optional positional embedding
- a stack of `TransformerBlock`
- returns full token sequence and CLS token

### 3.5 Attention pooling

- `models.heads.AttentionPool` is optionally enabled
- combines CLS token and pooled sequence tokens using a learned mix

### 3.6 Output heads

`EEGModel` produces:

- `task`: task classification logits via `TaskHead`
- `domain`: domain logits via `DomainHead`
- `features`: representation used for classification

The model also includes:

- gradient reversal layer (`GRL`) for adversarial domain adaptation support
- optional CNN domain head, though current training uses only task logits

## 4. Training flow

`training/run.py` controls the training loop:

1. Load data and determine subjects
2. Build model with default hyperparameters
3. For each selected subject:
   - build train/test loaders
   - run `train_one_subject(...)`
4. Save summary and history JSON files

### 4.1 Per-epoch training

- uses `torch.optim.AdamW`
- `torch.nn.CrossEntropyLoss`
- optional cosine learning-rate scheduler
- each epoch:
  - forward pass through model
  - compute task loss only
  - backward pass and optimizer step
  - evaluation on the validation loader (used for early stopping)
  - evaluation on the held-out test loader (not used for early stopping)

### 4.2 Evaluation

`evaluate(...)` uses the model's `task` logits and computes:

- accuracy
- Cohen's kappa

### 4.3 Within-subject cross-validation

When `--within_cv_folds` is greater than 1, each subject is evaluated using
stratified k-fold cross-validation. Metrics are averaged across folds per subject.

## 5. Outputs

Each run writes to `results/<protocol>_<dataset>_<timestamp>/`:

- `summary.json` with per-subject metrics and overall mean/std
- `history.json` with per-epoch training and evaluation curves
- `train.log` with detailed training logs

## 6. Maintenance guideline

Whenever the codebase changes, update this file to keep the architecture documentation aligned with the implementation.

- if the data loader changes, update the data flow section
- if model blocks are added/removed, update the model architecture section
- if training behavior changes, update the training flow section

## 7. Current design intent

This project is intentionally simplified to:

- rely on MOABB for EEG dataset loading and preprocessing
- load all motor imagery classes per dataset by default (no left/right filtering)
- use all EEG channels per dataset by default, with an optional common-channel mode
- use a single branded model pipeline
- support only supervised LOSO and within-subject runs
- avoid pretraining or domain adaptation complexity in the current main flow
