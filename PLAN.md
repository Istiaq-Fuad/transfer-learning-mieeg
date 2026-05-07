# EEG Motor Imagery Pipeline — Accuracy Improvement Plan

> Generated from architecture review of `training/run.py` and associated modules.
> Apply changes in tier order. Each tier builds on the previous.

---

## Tier 1 — High Impact, Low Implementation Cost

*Do these first. Minimal architectural changes, significant accuracy gains.*

---

### 1. Learning Rate Scheduler

**Root cause addressed:** Fixed-LR AdamW is a known failure mode for transformer-based models. Early attention weights destabilize without warmup; late training fails to escape sharp minima without decay.

**Implementation:**
```python
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=10)
cosine = CosineAnnealingLR(optimizer, T_max=epochs - 10, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[10])

# In training loop, after optimizer.step():
scheduler.step()
```

**Where to add:** `train_one_subject(...)` in `training/run.py`.

**Expected gain:** +2 to 5% accuracy on BNCI2014-001 LOSO.

---

### 2. Always-On Euclidean Alignment

**Root cause addressed:** Cross-subject covariance shift is the dominant source of variance in LOSO. Currently optional — this is a silent confound that must be eliminated.

**Implementation:**
- Remove the `optional` flag from `fit_euclidean_alignment(...)` calls in `data/loader.py`.
- Always fit alignment on training data, always apply to test data.
- Ensure alignment is fitted on training subjects only (not the test subject) to prevent data leakage.

```python
# In create_dataloaders(...):
ea_matrix = fit_euclidean_alignment(x_train)   # fit on train only
x_train = apply_euclidean_alignment(x_train, ea_matrix)
x_test  = apply_euclidean_alignment(x_test,  ea_matrix)  # apply same matrix to test
```

**Where to add:** `create_dataloaders(...)` and `create_within_subject_dataloaders(...)` in `data/loader.py`.

**Expected gain:** +3 to 8% accuracy on LOSO across datasets.

---

### 3. Label Smoothing

**Root cause addressed:** Motor imagery labels are inherently noisy — subjects do not always fully imagine the movement. Hard labels cause the model to become overconfident on noisy training samples.

**Implementation:**
```python
# Replace in training loop:
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
```

**Tuning:** Try `label_smoothing` in `{0.05, 0.10, 0.15, 0.20}`. Start with `0.1`.

**Where to add:** Loss instantiation in `train_one_subject(...)`.

**Expected gain:** +1 to 3% accuracy, larger improvement on noisier datasets (physionetmi).

---

### 4. Class-Weighted Loss

**Root cause addressed:** physionetmi and some cho2017 subjects have class imbalances. Unweighted loss silently biases predictions toward the majority class.

**Implementation:**
```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = torch.nn.CrossEntropyLoss(
    weight=weights_tensor,
    label_smoothing=0.1   # combine with item 3
)
```

**Where to add:** Before training loop in `train_one_subject(...)`.

**Expected gain:** +1 to 4% on imbalanced datasets, negligible cost on balanced ones.

---

## Tier 2 — Medium Impact, Moderate Implementation Cost

*Apply after Tier 1 is validated. Requires more code changes but uses existing architecture.*

---

### 5. Activate Gradient Reversal Layer (GRL) for LOSO

**Root cause addressed:** The GRL and `DomainHead` are already implemented but unused. Adversarial domain adaptation forces the CNN to learn subject-invariant features — the architecturally correct use of what is already there.

**Implementation:**
```python
# Anneal GRL lambda from 0 to target over training
def grl_lambda(epoch, total_epochs, lambda_max=0.3):
    p = epoch / total_epochs
    return lambda_max * (2 / (1 + np.exp(-10 * p)) - 1)

# In training loop:
lam = grl_lambda(epoch, total_epochs)
outputs = model(x, grl_lambda=lam)

task_loss   = criterion(outputs['task'], y)
domain_loss = F.cross_entropy(outputs['domain'], subject_ids)
loss = task_loss + 0.1 * domain_loss
```

**Notes:**
- Use subject ID as the domain label.
- Only activate for LOSO protocol — domain labels are meaningful there.
- Start with domain loss weight `0.1`; tune in `{0.05, 0.1, 0.2, 0.3}`.

**Where to add:** `train_one_subject(...)` and `EEGModel.forward(...)`.

**Expected gain:** +2 to 5% LOSO accuracy, larger on high-variance datasets.

---

### 6. Temporal Augmentation

**Root cause addressed:** EEG datasets have hundreds of trials per subject. Augmentation during training directly addresses overfitting without any preprocessing overhead.

**Three augmentations to implement:**

**a) Random temporal crop** — most impactful:
```python
def random_crop(x, crop_ratio=0.85):
    T = x.shape[-1]
    crop_len = int(T * crop_ratio)
    start = torch.randint(0, T - crop_len + 1, (1,)).item()
    return x[..., start:start + crop_len]
```

**b) Gaussian noise injection:**
```python
def add_noise(x, sigma=0.01):
    noise = torch.randn_like(x) * (x.std() * sigma)
    return x + noise
```

**c) Random channel dropout:**
```python
def channel_dropout(x, p=0.05):
    mask = torch.bernoulli(torch.ones(x.shape[1]) * (1 - p))
    return x * mask.unsqueeze(0).unsqueeze(-1)
```

Apply all three only during training, not evaluation. Compose in `EEGDataset.__getitem__` with a `training` flag, or in the training loop batch iteration.

**Expected gain:** +2 to 4% accuracy, most pronounced in within-subject evaluation.

---

### 7. Feature-Space Mixup

**Root cause addressed:** Input-space mixup is semantically ambiguous for EEG (interpolating raw signals is not meaningful). Feature-space mixup after the tokenizer regularizes the transformer more effectively than weight decay alone.

**Implementation:**
```python
def feature_mixup(features, labels, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(features.size(0))
    mixed = lam * features + (1 - lam) * features[idx]
    y_a, y_b = labels, labels[idx]
    return mixed, y_a, y_b, lam

def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

Apply after `EEGTokenizer` output, before `ViTEncoder`. Use during training only.

**Expected gain:** +1 to 3% accuracy; stronger regularization effect than weight decay alone.

---

### 8. 5-Fold Stratified Cross-Validation for Within-Subject

**Root cause addressed:** A single train/test split makes within-subject results highly variance-dependent. 5-fold CV produces a publishable, reproducible estimate required by most BCI venues.

**Implementation:**
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, test_idx) in enumerate(skf.split(x_subject, y_subject)):
    x_train, x_test = x_subject[train_idx], x_subject[test_idx]
    y_train, y_test = y_subject[train_idx], y_subject[test_idx]
    # ... build loaders, train, evaluate
    fold_results.append(evaluate(...))

subject_result = {
    'mean_accuracy': np.mean([r['accuracy'] for r in fold_results]),
    'std_accuracy':  np.std([r['accuracy'] for r in fold_results]),
    'mean_kappa':    np.mean([r['kappa'] for r in fold_results]),
}
```

**Where to add:** `create_within_subject_dataloaders(...)` in `data/loader.py`.

**Expected gain:** Not an accuracy gain per se — produces a reliable estimate. Removes single-split variance that can inflate or deflate reported numbers by ±5%.

---

## Tier 3 — High Impact, Higher Implementation Cost

*For ablation studies and the strongest possible reported results.*

---

### 9. Relative Positional Encoding (RoPE)

**Root cause addressed:** Fixed absolute positional embeddings assume EEG features occur at consistent absolute positions across trials. Motor imagery features vary in onset time by subject reaction speed. Relative encodings attend based on token distance, not position.

**Implementation:** Replace the existing positional embedding in `models/vit.py` with rotary positional embeddings (RoPE):

```python
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(q, k, cos, sin):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k
```

Pre-compute `cos` and `sin` tables in `ViTEncoder.__init__` and apply in each `TransformerBlock` attention computation.

**Reference:** Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021).

**Expected gain:** +1 to 3% on longer temporal sequences; larger benefit on physionetmi which has longer trial lengths.

---

### 10. Subject-Specific Calibration Layer

**Root cause addressed:** Even in LOSO, a few calibration trials from the test subject are typically available. A lightweight affine layer adapted on these trials closes a large portion of the domain gap without full retraining.

**Implementation:**
```python
class CalibrationLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(feature_dim))
        self.bias  = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, x):
        return x * self.scale + self.bias

# After LOSO training, adapt on N calibration trials from test subject:
cal_layer = CalibrationLayer(feature_dim).to(device)
cal_optimizer = torch.optim.Adam(cal_layer.parameters(), lr=1e-3)
for _ in range(50):   # few-shot fine-tune
    feats = model.get_features(x_cal)
    logits = model.task_head(cal_layer(feats))
    loss = criterion(logits, y_cal)
    loss.backward(); cal_optimizer.step(); cal_optimizer.zero_grad()
```

**Notes:** Use 10–20 calibration trials. Keep the base model frozen; only train the calibration layer. Report both with and without calibration in your paper.

**Expected gain:** +3 to 7% LOSO accuracy — one of the highest single-change gains possible.

---

### 11. Extended Evaluation Metrics

**Root cause addressed:** Accuracy and kappa alone cannot diagnose failure modes. Per-class F1 reveals class collapse; per-subject kappa variance reveals whether the model is consistently mediocre or wildly inconsistent (which requires different fixes).

**Implementation — add to `evaluate(...)` in `training/run.py`:**
```python
from sklearn.metrics import f1_score, confusion_matrix

def evaluate(model, loader, device):
    # ... existing accuracy/kappa computation ...
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return {
        'accuracy':  accuracy,
        'kappa':     kappa,
        'f1_macro':  f1_score(all_labels, all_preds, average='macro'),
        'f1_per_class': f1_score(all_labels, all_preds, average=None).tolist(),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
    }
```

Add to `summary.json`:
- Per-subject kappa with mean ± std
- Per-class F1 scores
- Confusion matrix per subject

**Expected gain:** Zero accuracy gain, but essential for paper diagnostics and reviewers.

---

### 12. Two-Phase LOSO Training

**Root cause addressed:** Training from scratch per subject ignores the cross-subject data already available in LOSO. Pre-training on all subjects first provides a better initialization for per-subject fine-tuning.

**Protocol:**

**Phase 1 — cross-subject pre-training:**
```python
# Use all subjects, no leave-one-out
x_all, y_all = load_all_subjects(dataset)
pretrain_loader = create_dataloader(x_all, y_all)
pretrain(model, pretrain_loader, epochs=50, lr=1e-3)
torch.save(model.state_dict(), 'pretrained_base.pt')
```

**Phase 2 — LOSO fine-tuning:**
```python
for test_subject in subjects:
    model.load_state_dict(torch.load('pretrained_base.pt'))  # reset to pretrained
    train_loader, test_loader = create_dataloaders(
        ..., loso_subject=test_subject
    )
    finetune(model, train_loader, epochs=20, lr=1e-4)   # lower LR for fine-tuning
    results[test_subject] = evaluate(model, test_loader)
```

**Notes:** Phase 2 uses a lower learning rate (1e-4 vs 1e-3). Freeze the CNN block during the first 5 fine-tuning epochs to prevent catastrophic forgetting, then unfreeze.

**Expected gain:** +3 to 6% LOSO accuracy. Largest benefit on datasets with many subjects (physionetmi).

---

## Implementation Schedule

| Week | Target | Changes |
|------|--------|---------|
| 1 | Tier 1 complete | Scheduler, always-on EA, label smoothing, class weights |
| 2 | Tier 1 baseline run | Re-run all experiments, establish new baseline |
| 3 | Tier 2 partial | GRL activation, temporal augmentation |
| 4 | Tier 2 complete | Mixup, 5-fold CV |
| 5 | Tier 2 baseline run | Ablation table: each Tier 2 change individually |
| 6–7 | Tier 3 | RoPE, calibration layer, two-phase training |
| 8 | Ablations + metrics | Extended metrics, confusion matrices, per-subject variance |

---

## Ablation Table Template

Run experiments in this order to produce a clean ablation table for the paper:

| Config | BNCI2014 Acc | BNCI2014 κ | physionetmi Acc | cho2017 Acc |
|--------|-------------|-----------|-----------------|-------------|
| Baseline (current) | — | — | — | — |
| + LR scheduler | | | | |
| + always EA | | | | |
| + label smoothing | | | | |
| + class weights (Tier 1 full) | | | | |
| + GRL domain loss | | | | |
| + augmentation | | | | |
| + mixup (Tier 2 full) | | | | |
| + RoPE | | | | |
| + calibration layer | | | | |
| + two-phase training (Tier 3 full) | | | | |

---

## Files to Modify

| File | Changes |
|------|---------|
| `training/run.py` | Scheduler, two-phase training, extended metrics |
| `training/utils.py` | Always-on EA, GRL lambda annealing |
| `data/loader.py` | Remove EA flag, add augmentation, 5-fold CV |
| `models/vit.py` | RoPE replacement |
| `models/model.py` | Expose calibration layer hook |
| `models/heads.py` | Calibration layer class |

---

## References

References are grouped by the plan section they support. All sourced from Consensus search, 2022–2025.

### Euclidean & Riemannian Alignment (§2 — Always-On EA)

- Wu, D. (2025). Revisiting Euclidean alignment for transfer learning in EEG-based brain–computer interfaces. *Journal of Neural Engineering, 22*. https://doi.org/10.1088/1741-2552/addd49
- Junqueira, B., Aristimunha, B., Chevallier, S., & De Camargo, R. (2024). A systematic evaluation of Euclidean alignment with deep learning for EEG decoding. *Journal of Neural Engineering, 21*. https://doi.org/10.1088/1741-2552/ad4f18
- Wang, H., Han, H., Gan, J., & Wang, H. (2024). Lightweight Source-Free Domain Adaptation Based on Adaptive Euclidean Alignment for Brain-Computer Interfaces. *IEEE Journal of Biomedical and Health Informatics, 29*, 909–922. https://doi.org/10.1109/jbhi.2024.3463737
- Zhuo, F., Zhang, X., Tang, F., Yu, Y., & Liu, L. (2024). Riemannian transfer learning based on log-Euclidean metric for EEG classification. *Frontiers in Neuroscience, 18*. https://doi.org/10.3389/fnins.2024.1381572
- Xie, C., Wang, L., Yang, J., & Guo, J. (2025). A subject transfer neural network fuses Generator and Euclidean alignment for EEG-based motor imagery classification. *Journal of Neuroscience Methods, 420*. https://doi.org/10.1016/j.jneumeth.2025.110483

### Domain Adversarial Training / GRL (§5 — Activate GRL)

- Chen, X., Wang, Z., & Wu, D. (2024). Alignment-Based Adversarial Training (ABAT) for Improving the Robustness and Accuracy of EEG-Based BCIs. *IEEE Transactions on Neural Systems and Rehabilitation Engineering, 32*, 1703–1714. https://doi.org/10.1109/tnsre.2024.3391936
- Liu, D., Zhang, J., Wu, H., Liu, S., & Long, J. (2022). Multi-Source Transfer Learning for EEG Classification Based on Domain Adversarial Neural Network. *IEEE Transactions on Neural Systems and Rehabilitation Engineering, 31*, 218–228. https://doi.org/10.1109/tnsre.2022.3219418

### Data Augmentation (§6 — Temporal Augmentation)

- Sun, C., & Mou, C. (2023). Survey on the research direction of EEG-based signal processing. *Frontiers in Neuroscience, 17*. https://doi.org/10.3389/fnins.2023.1203059
- Wang, X., Yang, R., & Huang, M. (2022). An Unsupervised Deep-Transfer-Learning-Based Motor Imagery EEG Classification Scheme for Brain–Computer Interface. *Sensors, 22*. https://doi.org/10.3390/s22062241

### Feature-Space Mixup (§7 — Mixup)

- Wei, F., Xu, X., Li, X., & Wu, X. (2024). BDAN-SPD: A Brain Decoding Adversarial Network Guided by Spatiotemporal Pattern Differences for Cross-Subject MI-BCI. *IEEE Transactions on Industrial Informatics, 20*, 14321–14329. https://doi.org/10.1109/tii.2024.3450010

### Relative Positional Encoding (§9 — RoPE)

- Huang, Z., & Chen, M. (2025). Optimizing the Learnable RoPE Theta Parameter in Transformers. *IEEE Access, 13*, 131271–131288. https://doi.org/10.1109/access.2025.3590604

### Subject-Specific Calibration (§10 — Calibration Layer)

- Wang, Y., Wang, J., Wang, W., Su, J., Bunterngchit, C., & Hou, Z. (2024). TFTL: A Task-Free Transfer Learning Strategy for EEG-Based Cross-Subject and Cross-Dataset Motor Imagery BCI. *IEEE Transactions on Biomedical Engineering, 72*, 810–821. https://doi.org/10.1109/tbme.2024.3474049
- Wang, Y., Wang, J., Wang, W., Su, J., & Hou, Z. (2023). Calibration-Free Transfer Learning for EEG-Based Cross-Subject Motor Imagery Classification. *2023 IEEE 19th International Conference on Automation Science and Engineering (CASE)*, 1–6. https://doi.org/10.1109/case56687.2023.10260440

### CNN-Transformer Hybrid Architecture (general — model context)

- Zhao, W., Jiang, X., Zhang, B., Xiao, S., & Weng, S. (2024). CTNet: a convolutional transformer network for EEG-based motor imagery classification. *Scientific Reports, 14*. https://doi.org/10.1038/s41598-024-71118-7

### Learning Rate Scheduling (§1 — LR Scheduler)

- Kai-Chen, M., Zhang, J., Huang, X., & Wang, M. (2025). Leveraging transformer models to predict cognitive impairment: accuracy, efficiency, and interpretability. *BMC Public Health, 25*. https://doi.org/10.1186/s12889-025-21762-z

---