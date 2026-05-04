from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, fields, replace
import logging
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
import warnings
import zipfile

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


@dataclass(frozen=True)
class MoabbLoadOptions:
    resample: int = 250
    tmin: float = 0.0
    tmax: float = 4.0
    fmin: float = 4.0
    fmax: float = 40.0
    subjects: list[int] | None = None
    max_subjects: int | None = None
    class_policy: str = "all"
    use_common_channels: bool = False
    mne_log_level: str | None = "ERROR"
    moabb_log_level: str | None = "ERROR"
    show_progress: bool = True
    subject_load_retries: int = 0
    redownload_on_failure: bool = True


@dataclass(frozen=True)
class DataLoaderOptions:
    batch_size: int = 32
    test_size: float = 0.2
    random_state: int = 42
    apply_euclidean_align: bool = True
    align_eps: float = 1e-6
    subject_balanced_sampling: bool = False
    drop_last_train: bool = False
    num_workers: int = 0
    seed: int | None = None
    deterministic: bool = True


def _resolve_moabb_load_options(
    options: MoabbLoadOptions | None,
    legacy_kwargs: dict[str, Any],
) -> MoabbLoadOptions:
    if options is None:
        resolved = MoabbLoadOptions()
    else:
        resolved = options

    if not legacy_kwargs:
        return resolved

    allowed = {f.name for f in fields(MoabbLoadOptions)}
    unknown = sorted(set(legacy_kwargs) - allowed)
    if unknown:
        unknown_text = ", ".join(unknown)
        raise TypeError(f"Unexpected load options: {unknown_text}")

    return replace(resolved, **legacy_kwargs)


def _resolve_data_loader_options(
    options: DataLoaderOptions | None,
    legacy_kwargs: dict[str, Any],
    allowed_fields: set[str],
) -> DataLoaderOptions:
    if options is None:
        resolved = DataLoaderOptions()
    else:
        resolved = options

    if not legacy_kwargs:
        return resolved

    unknown = sorted(set(legacy_kwargs) - allowed_fields)
    if unknown:
        unknown_text = ", ".join(unknown)
        raise TypeError(f"Unexpected dataloader options: {unknown_text}")

    return replace(resolved, **legacy_kwargs)


def split_eeg_data(
    x: np.ndarray,
    y: np.ndarray,
    subject_id: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    loso_subject: int | None = None,
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
    loso_subject: int | None = None,
    options: DataLoaderOptions | None = None,
    **legacy_options: Any,
) -> tuple[DataLoader, DataLoader]:
    opts = _resolve_data_loader_options(
        options,
        legacy_options,
        {
            "batch_size",
            "test_size",
            "random_state",
            "apply_euclidean_align",
            "align_eps",
            "subject_balanced_sampling",
            "drop_last_train",
            "num_workers",
            "seed",
            "deterministic",
        },
    )

    split = split_eeg_data(
        x=x,
        y=y,
        subject_id=subject_id,
        test_size=opts.test_size,
        random_state=opts.random_state,
        loso_subject=loso_subject,
    )

    if opts.apply_euclidean_align:
        whitening = fit_euclidean_alignment(split.train_dataset.x, eps=opts.align_eps)
        split.train_dataset.x = apply_euclidean_alignment(
            split.train_dataset.x, whitening
        )
        split.test_dataset.x = apply_euclidean_alignment(
            split.test_dataset.x, whitening
        )

    train_generator = None
    if opts.seed is not None and opts.deterministic:
        train_generator = build_torch_generator(opts.seed)

    train_sampler = None
    train_shuffle = True
    if opts.subject_balanced_sampling:
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
        batch_size=opts.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        drop_last=opts.drop_last_train,
        num_workers=opts.num_workers,
        generator=train_generator,
    )
    test_loader = DataLoader(
        split.test_dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.num_workers,
    )
    return train_loader, test_loader


def create_within_subject_dataloaders(
    x: np.ndarray,
    y: np.ndarray,
    subject_id: np.ndarray,
    target_subject: int,
    options: DataLoaderOptions | None = None,
    **legacy_options: Any,
) -> tuple[DataLoader, DataLoader]:
    """Create train/test loaders from one subject only."""
    opts = _resolve_data_loader_options(
        options,
        legacy_options,
        {
            "batch_size",
            "test_size",
            "random_state",
            "apply_euclidean_align",
            "align_eps",
            "subject_balanced_sampling",
            "drop_last_train",
            "num_workers",
            "seed",
            "deterministic",
        },
    )

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
        test_size=opts.test_size,
        random_state=opts.random_state,
        stratify=y_sub,
    )

    train_dataset = EEGDataset(x_sub[train_idx], y_sub[train_idx], s_sub[train_idx])
    test_dataset = EEGDataset(x_sub[test_idx], y_sub[test_idx], s_sub[test_idx])

    if opts.apply_euclidean_align:
        whitening = fit_euclidean_alignment(train_dataset.x, eps=opts.align_eps)
        train_dataset.x = apply_euclidean_alignment(train_dataset.x, whitening)
        test_dataset.x = apply_euclidean_alignment(test_dataset.x, whitening)

    train_generator = None
    if opts.seed is not None and opts.deterministic:
        train_generator = build_torch_generator(opts.seed)

    train_sampler = None
    train_shuffle = True
    if opts.subject_balanced_sampling:
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
        batch_size=opts.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        drop_last=opts.drop_last_train,
        num_workers=opts.num_workers,
        generator=train_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.num_workers,
    )
    return train_loader, test_loader


def _flatten_paths(value: Any) -> list[Path]:
    if isinstance(value, (str, Path)):
        return [Path(value)]
    if isinstance(value, (list, tuple, set)):
        paths: list[Path] = []
        for item in value:
            paths.extend(_flatten_paths(item))
        return paths
    return []


def _configure_moabb_data_path(data_path: str | None) -> str | None:
    if data_path is None:
        return None

    import os

    resolved = str(Path(data_path).expanduser().resolve())
    os.environ["MNE_DATA"] = resolved
    os.environ["MOABB_DATA_PATH"] = resolved
    os.environ["MNE_DATASETS_BNCI_PATH"] = resolved
    os.environ["MNE_DATASETS_EEGBCI_PATH"] = resolved
    os.environ["MNE_DATASETS_GIGADB_PATH"] = resolved
    os.environ["MNE_DATASETS_LEE2019_MI_PATH"] = resolved

    try:
        import mne

        mne.set_config("MNE_DATA", resolved, set_env=True)
    except Exception:
        pass

    return resolved


def _is_valid_mat_header(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            header = f.read(32)
    except OSError:
        return False
    return header.startswith(b"MATLAB 5.0 MAT-file")


def _is_cached_download_valid(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        if path.stat().st_size <= 0:
            return False
    except OSError:
        return False

    suffix = path.suffix.lower()
    if suffix == ".mat":
        return _is_valid_mat_header(path)
    if suffix == ".zip":
        try:
            with zipfile.ZipFile(path, "r") as zf:
                bad_crc = zf.testzip()
                return bad_crc is None
        except Exception:
            return False
    if suffix == ".edf":
        try:
            with path.open("rb") as f:
                return f.read(8) == b"0       "
        except OSError:
            return False
    if suffix == ".gdf":
        try:
            with path.open("rb") as f:
                return f.read(3) == b"GDF"
        except OSError:
            return False
    return True


def _infer_hash_algorithm(expected_hash: str) -> str | None:
    length = len(expected_hash)
    if length == 32:
        return "md5"
    if length == 40:
        return "sha1"
    if length == 64:
        return "sha256"
    return None


def _hash_matches(path: Path, known_hash: str | None) -> bool:
    if not known_hash:
        return True

    text = str(known_hash).strip()
    if not text or text.lower() == "unverified":
        return True

    expected = text.lower()
    if ":" in expected:
        algorithm, expected = expected.split(":", 1)
    else:
        inferred = _infer_hash_algorithm(expected)
        if inferred is None:
            return True
        algorithm = inferred

    try:
        import pooch
    except ModuleNotFoundError:
        return True

    try:
        computed = pooch.file_hash(path, alg=algorithm).lower()
    except Exception:
        return False
    return computed == expected


@contextmanager
def _cache_first_pooch_retrieve_context():
    try:
        import pooch
    except ModuleNotFoundError:
        yield {"enabled": False}
        return

    original_retrieve = pooch.retrieve
    allow_cache_reuse = {"enabled": True}

    def _patched_retrieve(url, known_hash, fname=None, path=None, **kwargs):
        if allow_cache_reuse["enabled"] and path is not None:
            candidate: Path | None = None
            if fname is not None:
                candidate = Path(path) / str(fname)
            else:
                parsed = urlparse(str(url))
                basename = Path(unquote(parsed.path)).name
                if basename:
                    candidate = Path(path) / basename

            if (
                candidate is not None
                and _is_cached_download_valid(candidate)
                and _hash_matches(candidate, known_hash)
            ):
                return str(candidate)

        return original_retrieve(url, known_hash, fname=fname, path=path, **kwargs)

    pooch.retrieve = _patched_retrieve
    try:
        yield allow_cache_reuse
    finally:
        pooch.retrieve = original_retrieve


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
        "feet": "feet",
        "both_feet": "feet",
        "foot": "feet",
        "tongue": "tongue",
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
            f"Dataset {dataset_name} has no left/right-hand trials. Available labels: {labels}"
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


def load_moabb_motor_imagery_dataset(
    dataset_name: str = "bnci2014_001",
    data_path: str | None = None,
    options: MoabbLoadOptions | None = None,
    **legacy_options: Any,
) -> Any:
    """
    Load EEG data from MOABB motor imagery datasets.

    Returns:
        x: (N, C, T) float32
        y: (N,) int64
        subject_id: (N,) int64
        subjects: available subject IDs
    """
    opts = _resolve_moabb_load_options(options, legacy_options)

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

    resolved_data_path = _configure_moabb_data_path(data_path)

    class_policy = str(opts.class_policy).strip().lower()
    channels = _COMMON_CHANNELS if opts.use_common_channels else None

    if class_policy == "left_right":
        if LeftRightImagery is not None:
            paradigm = LeftRightImagery(
                fmin=opts.fmin,
                fmax=opts.fmax,
                tmin=opts.tmin,
                tmax=opts.tmax,
                resample=opts.resample,
                channels=channels,
            )
        else:
            paradigm = MotorImagery(
                events=["left_hand", "right_hand"],
                n_classes=2,
                fmin=opts.fmin,
                fmax=opts.fmax,
                tmin=opts.tmin,
                tmax=opts.tmax,
                resample=opts.resample,
                channels=channels,
            )
    elif class_policy == "all":
        paradigm = MotorImagery(
            fmin=opts.fmin,
            fmax=opts.fmax,
            tmin=opts.tmin,
            tmax=opts.tmax,
            resample=opts.resample,
            channels=channels,
        )
    else:
        raise ValueError("Unsupported class_policy. Use one of: all, left_right")

    available_subjects = [int(s) for s in dataset.subject_list]
    selected_subjects = available_subjects
    if opts.subjects is not None:
        wanted = {int(s) for s in opts.subjects}
        selected_subjects = [s for s in available_subjects if s in wanted]
        if not selected_subjects:
            raise ValueError(f"No matching subjects found for dataset {dataset_name}")
    if opts.max_subjects is not None:
        if int(opts.max_subjects) <= 0:
            raise ValueError("max_subjects must be > 0 when provided")
        selected_subjects = selected_subjects[: int(opts.max_subjects)]
        if not selected_subjects:
            raise ValueError(f"No subjects selected for dataset {dataset_name}")
    if int(opts.subject_load_retries) < 0:
        raise ValueError("subject_load_retries must be >= 0")

    loader_logger = logging.getLogger(__name__)
    cache_reuse_control: dict[str, bool] = {"enabled": True}

    def _subject_data_paths(sid: int, force_update: bool = False) -> list[Path]:
        if not hasattr(dataset, "data_path"):
            return []
        kwargs: dict[str, Any] = {}
        if force_update:
            kwargs["force_update"] = True
        if resolved_data_path is not None:
            kwargs["path"] = resolved_data_path
        try:
            resolved = dataset.data_path(int(sid), **kwargs)
        except TypeError:
            kwargs.pop("force_update", None)
            resolved = dataset.data_path(int(sid), **kwargs)
        return _flatten_paths(resolved)

    def _force_redownload_subject(sid: int) -> None:
        if not opts.redownload_on_failure:
            return
        previous_reuse = cache_reuse_control.get("enabled", True)
        cache_reuse_control["enabled"] = False
        try:
            _subject_data_paths(int(sid), force_update=True)
        finally:
            cache_reuse_control["enabled"] = previous_reuse

    def _load_one_subject(sid: int) -> tuple[np.ndarray, np.ndarray, Any]:
        attempts = int(opts.subject_load_retries) + 1
        for attempt in range(attempts):
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    loaded = paradigm.get_data(dataset=dataset, subjects=[int(sid)])

                if caught:
                    warning_messages = [str(w.message) for w in caught]
                    first_message = warning_messages[0]

                    if attempt < (attempts - 1) and opts.redownload_on_failure:
                        loader_logger.warning(
                            "Warning for %s subject=%d: %s; forcing redownload and retrying",
                            dataset_name,
                            int(sid),
                            first_message,
                        )
                        _force_redownload_subject(int(sid))
                        continue

                    joined_messages = " | ".join(warning_messages)
                    raise RuntimeError(
                        "Subject load emitted warning(s) for "
                        f"{dataset_name} subject={int(sid)}: {joined_messages}"
                    )

                return loaded
            except Exception as exc:
                if attempt < (attempts - 1) and opts.redownload_on_failure:
                    loader_logger.warning(
                        "Error loading %s subject=%d: %s; forcing redownload and retrying",
                        dataset_name,
                        int(sid),
                        str(exc)[:200],
                    )
                    _force_redownload_subject(int(sid))
                    continue
                raise RuntimeError(
                    f"Failed loading {dataset_name} subject={int(sid)}: {exc}"
                ) from exc

        raise RuntimeError(
            f"Failed loading {dataset_name} subject={int(sid)} after retries"
        )

    with ExitStack() as stack:
        cache_reuse_control = stack.enter_context(_cache_first_pooch_retrieve_context())
        stack.enter_context(_mne_log_context(opts.mne_log_level))
        stack.enter_context(_moabb_log_context(opts.moabb_log_level))
        load_per_subject = (
            int(opts.subject_load_retries) > 0
            or bool(opts.redownload_on_failure)
            or (opts.show_progress and len(selected_subjects) > 1)
        )
        if load_per_subject:
            import pandas as pd

            x_parts: list[np.ndarray] = []
            y_parts: list[np.ndarray] = []
            meta_parts: list[Any] = []

            for sid in _progress_iter(
                selected_subjects,
                enabled=(opts.show_progress and len(selected_subjects) > 1),
                desc=f"Loading {dataset_name}",
            ):
                x_sid, y_sid, meta_sid = _load_one_subject(int(sid))
                x_parts.append(x_sid)
                y_parts.append(np.asarray(y_sid))
                meta_parts.append(meta_sid)

            if not x_parts:
                raise ValueError(f"Dataset {dataset_name} returned no trials")
            x = np.concatenate(x_parts, axis=0)
            y = np.concatenate(y_parts, axis=0)
            meta = pd.concat(meta_parts, axis=0, ignore_index=True)
        else:
            x, y, meta = paradigm.get_data(dataset=dataset, subjects=selected_subjects)

    if class_policy == "left_right":
        x, y, meta = _select_left_right_trials(
            x=x,
            y=y,
            meta=meta,
            dataset_name=dataset_name,
        )
    else:
        y = np.asarray([_canonical_motor_imagery_label(v) for v in y], dtype=object)

    # Common average reference keeps preprocessing consistent across datasets.
    x = x - x.mean(axis=1, keepdims=True)

    selected_subjects = sorted(meta["subject"].astype(int).unique().tolist())
    unique_labels = sorted(set([str(label) for label in y]))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_int = np.asarray([label_map[str(label)] for label in y], dtype=np.int64)
    s_int = meta["subject"].to_numpy(dtype=np.int64)

    x_out = x.astype(np.float32)
    return x_out, y_int, s_int, selected_subjects
