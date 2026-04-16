from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from training.utils import apply_euclidean_alignment, fit_euclidean_alignment
from utils.reproducibility import build_torch_generator


_COMMON_CHANNELS = [
    "FC3",
    "FC1",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
]


class EEGDataset(Dataset):
    """Simple EEG dataset returning x, y, and subject_id."""

    def __init__(self, x: np.ndarray, y: np.ndarray, subject_id: np.ndarray) -> None:
        if not (len(x) == len(y) == len(subject_id)):
            raise ValueError("x, y, and subject_id must have the same length")

        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.subject_id = torch.as_tensor(subject_id, dtype=torch.long)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx], self.subject_id[idx]


@dataclass
class SplitResult:
    train_dataset: EEGDataset
    test_dataset: EEGDataset


def split_eeg_data(
    x: np.ndarray,
    y: np.ndarray,
    subject_id: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    loso_subject: Optional[int] = None,
) -> SplitResult:
    """Split EEG data with either random split or LOSO strategy."""
    if loso_subject is not None:
        train_mask = subject_id != loso_subject
        test_mask = subject_id == loso_subject

        if test_mask.sum() == 0:
            raise ValueError(f"LOSO subject {loso_subject} not found in subject_id")

        train_dataset = EEGDataset(x[train_mask], y[train_mask], subject_id[train_mask])
        test_dataset = EEGDataset(x[test_mask], y[test_mask], subject_id[test_mask])
        return SplitResult(train_dataset=train_dataset, test_dataset=test_dataset)

    indices = np.arange(len(x))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    train_dataset = EEGDataset(x[train_idx], y[train_idx], subject_id[train_idx])
    test_dataset = EEGDataset(x[test_idx], y[test_idx], subject_id[test_idx])
    return SplitResult(train_dataset=train_dataset, test_dataset=test_dataset)


def create_dataloaders(
    x: np.ndarray,
    y: np.ndarray,
    subject_id: np.ndarray,
    batch_size: int = 32,
    test_size: float = 0.2,
    random_state: int = 42,
    loso_subject: Optional[int] = None,
    apply_euclidean_align: bool = True,
    align_eps: float = 1e-6,
    subject_balanced_sampling: bool = False,
    drop_last_train: bool = False,
    num_workers: int = 0,
    seed: Optional[int] = None,
    deterministic: bool = True,
) -> tuple[DataLoader, DataLoader]:
    split = split_eeg_data(
        x=x,
        y=y,
        subject_id=subject_id,
        test_size=test_size,
        random_state=random_state,
        loso_subject=loso_subject,
    )

    if apply_euclidean_align:
        whitening = fit_euclidean_alignment(split.train_dataset.x, eps=align_eps)
        split.train_dataset.x = apply_euclidean_alignment(
            split.train_dataset.x, whitening
        )
        split.test_dataset.x = apply_euclidean_alignment(
            split.test_dataset.x, whitening
        )

    train_generator = None
    if seed is not None and deterministic:
        train_generator = build_torch_generator(seed)

    train_sampler = None
    train_shuffle = True
    if subject_balanced_sampling:
        sid = split.train_dataset.subject_id
        unique_sid, counts = sid.unique(return_counts=True)
        if unique_sid.numel() > 1:
            count_map = {
                int(subject.item()): float(count.item())
                for subject, count in zip(unique_sid, counts)
            }
            weights = torch.tensor(
                [1.0 / count_map[int(s.item())] for s in sid],
                dtype=torch.double,
            )
            train_sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(split.train_dataset),
                replacement=True,
                generator=train_generator,
            )
            train_shuffle = False

    train_loader = DataLoader(
        split.train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        drop_last=drop_last_train,
        num_workers=num_workers,
        generator=train_generator,
    )
    test_loader = DataLoader(
        split.test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def create_within_subject_dataloaders(
    x: np.ndarray,
    y: np.ndarray,
    subject_id: np.ndarray,
    target_subject: int,
    batch_size: int = 32,
    test_size: float = 0.2,
    random_state: int = 42,
    apply_euclidean_align: bool = True,
    align_eps: float = 1e-6,
    subject_balanced_sampling: bool = False,
    drop_last_train: bool = False,
    num_workers: int = 0,
    seed: Optional[int] = None,
    deterministic: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Create train/test loaders from one subject only."""
    mask = subject_id == target_subject
    if int(mask.sum()) == 0:
        raise ValueError(f"Subject {target_subject} not found")

    x_sub = x[mask]
    y_sub = y[mask]
    s_sub = subject_id[mask]

    if np.unique(y_sub).shape[0] < 2:
        raise ValueError(
            f"Subject {target_subject} has fewer than 2 classes after filtering"
        )

    indices = np.arange(len(x_sub))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y_sub,
    )

    train_dataset = EEGDataset(x_sub[train_idx], y_sub[train_idx], s_sub[train_idx])
    test_dataset = EEGDataset(x_sub[test_idx], y_sub[test_idx], s_sub[test_idx])

    if apply_euclidean_align:
        whitening = fit_euclidean_alignment(train_dataset.x, eps=align_eps)
        train_dataset.x = apply_euclidean_alignment(train_dataset.x, whitening)
        test_dataset.x = apply_euclidean_alignment(test_dataset.x, whitening)

    train_generator = None
    if seed is not None and deterministic:
        train_generator = build_torch_generator(seed)

    train_sampler = None
    train_shuffle = True
    if subject_balanced_sampling:
        sid = train_dataset.subject_id
        unique_sid, counts = sid.unique(return_counts=True)
        if unique_sid.numel() > 1:
            count_map = {
                int(subject.item()): float(count.item())
                for subject, count in zip(unique_sid, counts)
            }
            weights = torch.tensor(
                [1.0 / count_map[int(s.item())] for s in sid],
                dtype=torch.double,
            )
            train_sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(train_dataset),
                replacement=True,
                generator=train_generator,
            )
            train_shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        drop_last=drop_last_train,
        num_workers=num_workers,
        generator=train_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def create_loso_domain_adaptation_dataloaders(
    x: np.ndarray,
    y: np.ndarray,
    subject_id: np.ndarray,
    target_subject: int,
    batch_size: int = 32,
    apply_euclidean_align: bool = True,
    align_eps: float = 1e-6,
    drop_last_train: bool = False,
    num_workers: int = 0,
    seed: Optional[int] = None,
    deterministic: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build LOSO loaders for adversarial DA.

    Source loader: all non-target-subject trials with labels.
    Target train loader: target-subject trials (labels ignored during training).
    Test loader: target-subject trials for evaluation.
    """
    source_mask = subject_id != target_subject
    target_mask = subject_id == target_subject

    if int(source_mask.sum()) == 0:
        raise ValueError(
            f"No source samples found when target_subject={target_subject}"
        )
    if int(target_mask.sum()) == 0:
        raise ValueError(f"Target subject {target_subject} not found")

    source_dataset = EEGDataset(
        x[source_mask],
        y[source_mask],
        np.zeros(int(source_mask.sum()), dtype=np.int64),
    )
    target_train_dataset = EEGDataset(
        x[target_mask],
        y[target_mask],
        np.ones(int(target_mask.sum()), dtype=np.int64),
    )
    target_test_dataset = EEGDataset(
        x[target_mask],
        y[target_mask],
        np.ones(int(target_mask.sum()), dtype=np.int64),
    )

    if apply_euclidean_align:
        whitening = fit_euclidean_alignment(source_dataset.x, eps=align_eps)
        source_dataset.x = apply_euclidean_alignment(source_dataset.x, whitening)
        target_train_dataset.x = apply_euclidean_alignment(
            target_train_dataset.x, whitening
        )
        target_test_dataset.x = apply_euclidean_alignment(
            target_test_dataset.x, whitening
        )

    generator = None
    if seed is not None and deterministic:
        generator = build_torch_generator(seed)

    source_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last_train,
        num_workers=num_workers,
        generator=generator,
    )
    target_train_loader = DataLoader(
        target_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last_train,
        num_workers=num_workers,
        generator=generator,
    )
    target_test_loader = DataLoader(
        target_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return source_loader, target_train_loader, target_test_loader


def create_within_subject_domain_adaptation_dataloaders(
    x: np.ndarray,
    y: np.ndarray,
    subject_id: np.ndarray,
    session_id: np.ndarray,
    target_subject: int,
    target_session: Optional[int] = None,
    batch_size: int = 32,
    target_test_size: float = 0.2,
    random_state: int = 42,
    apply_euclidean_align: bool = True,
    align_eps: float = 1e-6,
    drop_last_train: bool = False,
    num_workers: int = 0,
    seed: Optional[int] = None,
    deterministic: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Build within-subject DA loaders by splitting domains with session IDs.

    Source loader: selected subject, all sessions except target session.
    Target train loader: target session (unlabeled in training).
    Test loader: held-out subset from target session.
    """
    mask = subject_id == target_subject
    if int(mask.sum()) == 0:
        raise ValueError(f"Subject {target_subject} not found")

    x_sub = x[mask]
    y_sub = y[mask]
    s_sub = session_id[mask]

    sessions = np.unique(s_sub)
    if sessions.shape[0] < 2:
        raise ValueError(
            f"Subject {target_subject} needs at least 2 sessions for DA, got {sessions.shape[0]}"
        )

    selected_target_session = int(sessions[-1])
    if target_session is not None:
        if int(target_session) not in {int(v) for v in sessions.tolist()}:
            raise ValueError(
                f"target_session={target_session} not found for subject {target_subject}"
            )
        selected_target_session = int(target_session)

    source_mask = s_sub != selected_target_session
    target_mask = s_sub == selected_target_session

    if np.unique(y_sub[source_mask]).shape[0] < 2:
        raise ValueError(
            f"Source sessions for subject {target_subject} have fewer than 2 classes"
        )
    if np.unique(y_sub[target_mask]).shape[0] < 2:
        raise ValueError(
            f"Target session for subject {target_subject} has fewer than 2 classes"
        )

    target_indices = np.where(target_mask)[0]
    target_train_idx, target_test_idx = train_test_split(
        target_indices,
        test_size=target_test_size,
        random_state=random_state,
        stratify=y_sub[target_mask],
    )

    source_dataset = EEGDataset(
        x_sub[source_mask],
        y_sub[source_mask],
        np.zeros(int(source_mask.sum()), dtype=np.int64),
    )
    target_train_dataset = EEGDataset(
        x_sub[target_train_idx],
        y_sub[target_train_idx],
        np.ones(target_train_idx.shape[0], dtype=np.int64),
    )
    target_test_dataset = EEGDataset(
        x_sub[target_test_idx],
        y_sub[target_test_idx],
        np.ones(target_test_idx.shape[0], dtype=np.int64),
    )

    if apply_euclidean_align:
        whitening = fit_euclidean_alignment(source_dataset.x, eps=align_eps)
        source_dataset.x = apply_euclidean_alignment(source_dataset.x, whitening)
        target_train_dataset.x = apply_euclidean_alignment(
            target_train_dataset.x, whitening
        )
        target_test_dataset.x = apply_euclidean_alignment(
            target_test_dataset.x, whitening
        )

    generator = None
    if seed is not None and deterministic:
        generator = build_torch_generator(seed)

    source_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last_train,
        num_workers=num_workers,
        generator=generator,
    )
    target_train_loader = DataLoader(
        target_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last_train,
        num_workers=num_workers,
        generator=generator,
    )
    target_test_loader = DataLoader(
        target_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return (
        source_loader,
        target_train_loader,
        target_test_loader,
        selected_target_session,
    )


def _encode_meta_column(values: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    labels = [str(v) for v in values.tolist()]
    uniq = sorted(set(labels))
    mapping = {label: idx for idx, label in enumerate(uniq)}
    encoded = np.asarray([mapping[label] for label in labels], dtype=np.int64)
    return encoded, mapping


def load_moabb_motor_imagery_dataset(
    dataset_name: str = "bnci2014_001",
    data_path: str | None = None,
    resample: int = 128,
    tmin: float = 0.0,
    tmax: float = 4.0,
    fmin: float = 4.0,
    fmax: float = 40.0,
    subjects: Optional[list[int]] = None,
    include_metadata: bool = False,
) -> Any:
    """
    Load EEG data from MOABB MotorImagery paradigms.

    Returns:
        x: (N, C, T) float32
        y: (N,) int64
        subject_id: (N,) int64
        subjects: available subject IDs
    """
    from moabb.datasets import BNCI2014_001, Cho2017, Lee2019_MI, PhysionetMI
    from moabb.paradigms import MotorImagery

    name = dataset_name.lower()
    if name in {"bnci2014_001", "bnci", "bci"}:
        dataset = BNCI2014_001()
    elif name in {"physionetmi", "physionet"}:
        dataset = PhysionetMI(imagined=True, executed=False)
    elif name in {"cho2017", "cho"}:
        dataset = Cho2017()
    elif name in {"lee2019_mi", "lee2019", "lee"}:
        dataset = Lee2019_MI()
    else:
        raise ValueError(
            "Unsupported dataset_name. Use one of: bnci2014_001, physionetmi, cho2017, lee2019_mi"
        )

    if data_path:
        import os

        os.environ["MNE_DATA"] = data_path

    paradigm = MotorImagery(
        events=["left_hand", "right_hand"],
        n_classes=2,
        fmin=fmin,
        fmax=fmax,
        tmin=tmin,
        tmax=tmax,
        resample=resample,
        channels=_COMMON_CHANNELS,
    )

    available_subjects = [int(s) for s in dataset.subject_list]
    selected_subjects = available_subjects
    if subjects is not None:
        wanted = {int(s) for s in subjects}
        selected_subjects = [s for s in available_subjects if s in wanted]
        if not selected_subjects:
            raise ValueError(f"No matching subjects found for dataset {dataset_name}")

    x, y, meta = paradigm.get_data(dataset=dataset, subjects=selected_subjects)

    label_map = {label: idx for idx, label in enumerate(sorted(np.unique(y).tolist()))}
    y_int = np.asarray([label_map[label] for label in y], dtype=np.int64)
    s_int = meta["subject"].to_numpy(dtype=np.int64)

    if not include_metadata:
        return x.astype(np.float32), y_int, s_int, selected_subjects

    metadata: dict[str, Any] = {}
    if "session" in meta.columns:
        session_id, session_map = _encode_meta_column(meta["session"].to_numpy())
        metadata["session_id"] = session_id
        metadata["session_map"] = session_map
    if "run" in meta.columns:
        run_id, run_map = _encode_meta_column(meta["run"].to_numpy())
        metadata["run_id"] = run_id
        metadata["run_map"] = run_map

    return x.astype(np.float32), y_int, s_int, selected_subjects, metadata


def subsample_train_trials_per_subject_class(
    x: np.ndarray,
    y: np.ndarray,
    subject_id: np.ndarray,
    fraction: float,
    random_state: int = 42,
    min_trials_per_class: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Subsample trials with stratification per subject and per class.

    Args:
        x: (N, C, T)
        y: (N,)
        subject_id: (N,)
        fraction: keep ratio in (0, 1]
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1]")
    if fraction >= 1.0:
        return x, y, subject_id

    rng = np.random.default_rng(random_state)
    keep_indices: list[np.ndarray] = []

    for sid in np.unique(subject_id):
        sid_mask = subject_id == sid
        y_sid = y[sid_mask]
        sid_indices = np.where(sid_mask)[0]

        for cls in np.unique(y_sid):
            cls_local = np.where(y_sid == cls)[0]
            cls_global = sid_indices[cls_local]
            n_total = cls_global.shape[0]
            n_keep = max(min_trials_per_class, int(round(n_total * fraction)))
            n_keep = min(n_keep, n_total)
            chosen = rng.choice(cls_global, size=n_keep, replace=False)
            keep_indices.append(np.sort(chosen))

    if not keep_indices:
        raise ValueError("No samples selected after subsampling")

    final_idx = np.concatenate(keep_indices)
    final_idx.sort()
    return x[final_idx], y[final_idx], subject_id[final_idx]
