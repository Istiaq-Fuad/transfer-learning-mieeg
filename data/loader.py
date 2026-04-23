from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
import logging

import pooch
from pathlib import Path
_orig_retrieve = pooch.retrieve

def _patched_pooch_retrieve(url, known_hash, fname=None, path=None, **kwargs):
    if path is not None and fname is not None:
        p = Path(path) / fname
        if p.exists() and p.stat().st_size > 1000000:
            return str(p)
    return _orig_retrieve(url, known_hash, fname=fname, path=path, **kwargs)

pooch.retrieve = _patched_pooch_retrieve

from pathlib import Path
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


def _mne_log_context(level: str | None):
    if level is None:
        return nullcontext()
    try:
        import mne

    except ModuleNotFoundError:
        return nullcontext()
    return mne.use_log_level(level, add_frames=False)


def _progress_iter(values: list[int], enabled: bool, desc: str):
    if not enabled:
        return values
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        return values
    return tqdm(values, desc=desc, unit="subject")


def _parse_log_level(level: str) -> int:
    value = getattr(logging, level.upper(), None)
    if not isinstance(value, int):
        raise ValueError(f"Unsupported log level: {level}")
    return value


@contextmanager
def _moabb_log_context(level: str | None):
    if level is None:
        yield
        return

    parsed = _parse_log_level(level)
    logger_names = ("moabb", "moabb.datasets", "moabb.datasets.gigadb")
    loggers = [logging.getLogger(name) for name in logger_names]
    previous_levels = [logger.level for logger in loggers]

    for logger in loggers:
        logger.setLevel(parsed)

    try:
        yield
    finally:
        for logger, previous in zip(loggers, previous_levels):
            logger.setLevel(previous)


def _canonical_motor_imagery_label(label: Any) -> str:
    text = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "left": "left_hand",
        "left_hand": "left_hand",
        "right": "right_hand",
        "right_hand": "right_hand",
    }
    return aliases.get(text, text)


def _select_left_right_trials(
    x: np.ndarray,
    y: np.ndarray,
    meta: Any,
    dataset_name: str,
) -> tuple[np.ndarray, np.ndarray, Any]:
    canonical = np.asarray([_canonical_motor_imagery_label(v) for v in y], dtype=object)
    keep = np.isin(canonical, ["left_hand", "right_hand"])
    if int(keep.sum()) == 0:
        labels = sorted(set(canonical.tolist()))
        raise ValueError(
            f"Dataset {dataset_name} has no left/right-hand trials in this configuration. Available labels: {labels}"
        )

    x_lr = x[keep]
    y_lr = canonical[keep]
    meta_lr = meta.iloc[np.where(keep)[0]].reset_index(drop=True)

    present = sorted(set(y_lr.tolist()))
    if present != ["left_hand", "right_hand"]:
        raise ValueError(
            f"Dataset {dataset_name} does not contain both left and right classes after filtering: {present}"
        )

    return x_lr, y_lr, meta_lr


def _canonicalize_labels(y: np.ndarray) -> np.ndarray:
    return np.asarray([_canonical_motor_imagery_label(v) for v in y], dtype=object)


def _flatten_paths(value: Any) -> list[Path]:
    if isinstance(value, (str, Path)):
        return [Path(value)]
    if isinstance(value, (list, tuple, set)):
        paths: list[Path] = []
        for item in value:
            paths.extend(_flatten_paths(item))
        return paths
    return []


def _is_valid_mat_header(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            header = f.read(32)
    except OSError:
        return False
    return header.startswith(b"MATLAB 5.0 MAT-file")


def load_moabb_motor_imagery_dataset(
    dataset_name: str = "bnci2014_001",
    data_path: str | None = None,
    resample: int = 128,
    tmin: float = 0.0,
    tmax: float = 4.0,
    fmin: float = 4.0,
    fmax: float = 40.0,
    subjects: Optional[list[int]] = None,
    max_subjects: Optional[int] = None,
    include_metadata: bool = False,
    mne_log_level: str | None = "WARNING",
    moabb_log_level: str | None = "ERROR",
    class_policy: str = "left_right",
    show_progress: bool = True,
    skip_failed_subjects: bool = False,
    subject_load_retries: int = 0,
    redownload_on_failure: bool = True,
    redownload_once_per_subject: bool = True,
    skip_known_failed_subjects: bool = False,
) -> Any:
    """
    Load EEG data from MOABB MotorImagery paradigms.

    Returns:
        x: (N, C, T) float32
        y: (N,) int64
        subject_id: (N,) int64
        subjects: available subject IDs

    class_policy:
        - "left_right": keep only left/right classes
        - "all_mi": keep all motor-imagery classes from paradigm output

    moabb_log_level:
        - controls MOABB logger verbosity during dataset load
        - default "ERROR" hides non-critical warnings

    show_progress:
        - when True, shows subject-level progress bar while loading

    skip_failed_subjects:
        - when True, subject-level load failures are logged and skipped

    subject_load_retries:
        - retries per subject before skipping/failing

    redownload_on_failure:
        - when True, try force-updating failing subject files before retry

    redownload_once_per_subject:
        - when True, each dataset/subject pair is force-redownloaded at most once
          across runs (marker stored under MNE_DATA/.redownload_attempts)

    skip_known_failed_subjects:
        - when True, subjects marked as failed in a previous run are skipped
          immediately to avoid repeated download attempts

    max_subjects:
        - if set, cap selected subjects to the first N available IDs
    """
    from moabb.datasets import BNCI2014_001, Cho2017, Lee2019_MI, PhysionetMI
    from moabb.paradigms import MotorImagery

    try:
        from moabb.paradigms import LeftRightImagery
    except ImportError:
        LeftRightImagery = None

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

    if class_policy == "all_mi":
        paradigm = MotorImagery(
            fmin=fmin,
            fmax=fmax,
            tmin=tmin,
            tmax=tmax,
            resample=resample,
            channels=_COMMON_CHANNELS,
        )
    elif LeftRightImagery is not None:
        paradigm = LeftRightImagery(
            fmin=fmin,
            fmax=fmax,
            tmin=tmin,
            tmax=tmax,
            resample=resample,
            channels=_COMMON_CHANNELS,
        )
    else:
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
    if max_subjects is not None:
        if int(max_subjects) <= 0:
            raise ValueError("max_subjects must be > 0 when provided")
        selected_subjects = selected_subjects[: int(max_subjects)]
        if not selected_subjects:
            raise ValueError(f"No subjects selected for dataset {dataset_name}")
    if int(subject_load_retries) < 0:
        raise ValueError("subject_load_retries must be >= 0")

    loader_logger = logging.getLogger(__name__)
    dataset_key = str(getattr(dataset, "code", dataset_name)).lower()

    marker_root = (
        Path(data_path) / ".loader_markers"
        if data_path
        else Path.home() / ".cache" / "transfer-learning-bci-loader"
    )
    redownload_marker_dir = marker_root / "redownload_attempts"
    failed_marker_dir = marker_root / "failed_subjects"

    def _redownload_marker_path(sid: int) -> Path | None:
        return redownload_marker_dir / f"{dataset_key}_subject_{int(sid)}.marker"

    def _failed_marker_path(sid: int) -> Path:
        return failed_marker_dir / f"{dataset_key}_subject_{int(sid)}.marker"

    def _is_redownloadable_error(exc: Exception) -> bool:
        if isinstance(exc, OSError):
            return True
        text = str(exc).lower()
        return (
            "input/output error" in text
            or "i/o error" in text
            or "file_hash" in text
            or "errno 5" in text
            or "unknown mat file type" in text
            or "expecting matrix here" in text
            or "mat file appears to be truncated" in text
            or "not a mat file" in text
        )

    def _subject_data_paths(sid: int, force_update: bool = False) -> list[Path]:
        if not hasattr(dataset, "data_path"):
            return []
        kwargs: dict[str, Any] = {"force_update": bool(force_update)}
        if data_path is not None:
            kwargs["path"] = data_path
        try:
            resolved = dataset.data_path(int(sid), **kwargs)
        except TypeError:
            # Older MOABB signatures may reject some kwargs.
            kwargs.pop("force_update", None)
            resolved = dataset.data_path(int(sid), **kwargs)
        return _flatten_paths(resolved)

    def _has_invalid_mat_files(sid: int) -> bool:
        paths = _subject_data_paths(int(sid), force_update=False)
        for p in paths:
            if p.suffix.lower() != ".mat":
                continue
            if not p.exists() or not _is_valid_mat_header(p):
                return True
        return False

    def _force_redownload_subject(sid: int) -> None:
        if not redownload_on_failure:
            return
        marker = _redownload_marker_path(int(sid))
        if marker is not None and marker.exists():
            loader_logger.warning(
                "Skipping force redownload for %s subject=%d (already attempted before)",
                dataset_name,
                int(sid),
            )
            return
        _subject_data_paths(int(sid), force_update=True)
        if marker is not None:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("attempted\n", encoding="utf-8")

    def _mark_failed_subject(sid: int, error_text: str) -> None:
        marker = _failed_marker_path(int(sid))
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(error_text + "\n", encoding="utf-8")

    def _clear_failed_subject_marker(sid: int) -> None:
        marker = _failed_marker_path(int(sid))
        if marker.exists():
            marker.unlink()

    def _load_one_subject(sid: int) -> tuple[np.ndarray, np.ndarray, Any]:
        attempts = int(subject_load_retries) + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            if attempt == 0 and redownload_on_failure:
                try:
                    if _has_invalid_mat_files(int(sid)):
                        loader_logger.warning(
                            "Detected invalid MAT file for %s subject=%d; forcing redownload",
                            dataset_name,
                            int(sid),
                        )
                        _force_redownload_subject(int(sid))
                except Exception as precheck_exc:
                    loader_logger.warning(
                        "MAT precheck failed for %s subject=%d: %s",
                        dataset_name,
                        int(sid),
                        str(precheck_exc),
                    )
            try:
                return paradigm.get_data(dataset=dataset, subjects=[int(sid)])
            except Exception as exc:  # pragma: no cover
                last_error = exc
                has_retry = attempt < (attempts - 1)
                if has_retry and _is_redownloadable_error(exc):
                    try:
                        _force_redownload_subject(int(sid))
                        loader_logger.warning(
                            "Retrying %s subject=%d after force redownload",
                            dataset_name,
                            int(sid),
                        )
                    except Exception as red_exc:
                        loader_logger.warning(
                            "Redownload failed for %s subject=%d: %s",
                            dataset_name,
                            int(sid),
                            str(red_exc),
                        )
        if last_error is None:
            raise RuntimeError(f"Unknown subject load error for {sid}")
        raise last_error

    with ExitStack() as stack:
        stack.enter_context(_mne_log_context(mne_log_level))
        stack.enter_context(_moabb_log_context(moabb_log_level))
        load_per_subject = (
            skip_failed_subjects
            or (show_progress and len(selected_subjects) > 1)
            or int(subject_load_retries) > 0
            or bool(redownload_on_failure)
            or bool(skip_known_failed_subjects)
        )
        if load_per_subject:
            import pandas as pd

            x_parts: list[np.ndarray] = []
            y_parts: list[np.ndarray] = []
            meta_parts: list[Any] = []
            failed_subjects: list[int] = []

            for sid in _progress_iter(
                selected_subjects,
                enabled=(show_progress and len(selected_subjects) > 1),
                desc=f"Loading {dataset_name}",
            ):
                if skip_known_failed_subjects and _failed_marker_path(int(sid)).exists():
                    failed_subjects.append(int(sid))
                    loader_logger.warning(
                        "Skipping %s subject=%d due to previous failure marker",
                        dataset_name,
                        int(sid),
                    )
                    continue
                try:
                    x_sid, y_sid, meta_sid = _load_one_subject(int(sid))
                except Exception as exc:  # pragma: no cover
                    if not skip_failed_subjects:
                        raise
                    failed_subjects.append(int(sid))
                    _mark_failed_subject(int(sid), str(exc))
                    loader_logger.warning(
                        "Skipping %s subject=%d due to load error: %s",
                        dataset_name,
                        int(sid),
                        str(exc),
                    )
                    continue
                _clear_failed_subject_marker(int(sid))
                x_parts.append(x_sid)
                y_parts.append(np.asarray(y_sid))
                meta_parts.append(meta_sid)

            if not x_parts:
                if failed_subjects:
                    raise ValueError(
                        f"Dataset {dataset_name} failed for all selected subjects: {failed_subjects}"
                    )
                raise ValueError(f"Dataset {dataset_name} returned no trials")
            x = np.concatenate(x_parts, axis=0)
            y = np.concatenate(y_parts, axis=0)
            meta = pd.concat(meta_parts, axis=0, ignore_index=True)
            if failed_subjects:
                loader_logger.warning(
                    "Loaded %s with %d skipped subjects: %s",
                    dataset_name,
                    len(failed_subjects),
                    failed_subjects,
                )
        else:
            x, y, meta = paradigm.get_data(dataset=dataset, subjects=selected_subjects)

    if class_policy == "left_right":
        x, y, meta = _select_left_right_trials(
            x=x,
            y=y,
            meta=meta,
            dataset_name=dataset_name,
        )
    elif class_policy == "all_mi":
        y = _canonicalize_labels(y)
        if int(x.shape[0]) == 0:
            raise ValueError(f"Dataset {dataset_name} has no trials after loading")
    else:
        raise ValueError(
            f"Unsupported class_policy={class_policy}. Use one of: left_right, all_mi"
        )

    selected_subjects = sorted(meta["subject"].astype(int).unique().tolist())
    if class_policy == "left_right":
        label_map = {"left_hand": 0, "right_hand": 1}
    else:
        classes = sorted(set(str(label) for label in y.tolist()))
        label_map = {label: idx for idx, label in enumerate(classes)}
    y_int = np.asarray([label_map[str(label)] for label in y], dtype=np.int64)
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
