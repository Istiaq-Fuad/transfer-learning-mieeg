#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.signal import welch

try:
    from data.loader import MoabbLoadOptions, load_moabb_motor_imagery_dataset
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from data.loader import MoabbLoadOptions, load_moabb_motor_imagery_dataset


@dataclass
class SubjectQuality:
    subject_id: int
    n_trials: int
    median_rms: float
    median_ptp: float
    artifact_rate: float
    alpha_ratio: float
    hf_ratio: float
    snr_proxy: float
    quality_score: float = 0.0


def _bandpower_from_psd(
    freqs: np.ndarray, psd: np.ndarray, low: float, high: float
) -> np.ndarray:
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return np.zeros(psd.shape[0], dtype=np.float64)
    return np.trapezoid(psd[:, mask], freqs[mask], axis=1)


def _safe_ratio(num: np.ndarray, den: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return num / np.maximum(den, eps)


def _subject_quality_metrics(
    x_subject: np.ndarray, sfreq: float, subject_id: int
) -> SubjectQuality:
    # x_subject shape: (N_trials, C, T)
    n_trials = int(x_subject.shape[0])

    # Trial-level amplitude metrics.
    rms = np.sqrt(np.mean(np.square(x_subject), axis=(1, 2)))
    ptp = np.ptp(x_subject, axis=2).mean(axis=1)

    # Mark likely artifact trials using robust MAD z-score on trial peak-to-peak amplitude.
    med = float(np.median(ptp))
    mad = float(np.median(np.abs(ptp - med)))
    robust_scale = 1.4826 * mad + 1e-12
    robust_z = np.abs((ptp - med) / robust_scale)
    artifact_rate = float(np.mean(robust_z > 3.5))

    # Compute a PSD per trial by averaging channel PSDs.
    # Input loader already applies filtering/resampling; this is a quality proxy, not raw QC.
    trial_psd: list[np.ndarray] = []
    out_freqs: np.ndarray | None = None
    nperseg = min(256, x_subject.shape[-1])
    for trial in x_subject:
        freqs, psd_ch = welch(
            trial,
            fs=sfreq,
            nperseg=nperseg,
            axis=-1,
        )
        out_freqs = freqs
        trial_psd.append(np.mean(psd_ch, axis=0))

    if out_freqs is None:
        raise ValueError(f"No PSD computed for subject {subject_id}")

    psd = np.asarray(trial_psd, dtype=np.float64)
    total_4_40 = _bandpower_from_psd(out_freqs, psd, 4.0, 40.0)
    alpha_8_13 = _bandpower_from_psd(out_freqs, psd, 8.0, 13.0)
    sensorimotor_8_30 = _bandpower_from_psd(out_freqs, psd, 8.0, 30.0)
    hf_30_40 = _bandpower_from_psd(out_freqs, psd, 30.0, 40.0)

    alpha_ratio = float(np.median(_safe_ratio(alpha_8_13, total_4_40)))
    hf_ratio = float(np.median(_safe_ratio(hf_30_40, total_4_40)))
    snr_proxy = float(np.median(_safe_ratio(sensorimotor_8_30, hf_30_40)))

    return SubjectQuality(
        subject_id=subject_id,
        n_trials=n_trials,
        median_rms=float(np.median(rms)),
        median_ptp=float(np.median(ptp)),
        artifact_rate=artifact_rate,
        alpha_ratio=alpha_ratio,
        hf_ratio=hf_ratio,
        snr_proxy=snr_proxy,
    )


def _minmax(values: np.ndarray) -> np.ndarray:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        return np.full_like(values, 0.5, dtype=np.float64)
    return (values - vmin) / (vmax - vmin)


def _attach_quality_scores(rows: list[SubjectQuality]) -> list[SubjectQuality]:
    alpha = np.asarray([r.alpha_ratio for r in rows], dtype=np.float64)
    hf = np.asarray([r.hf_ratio for r in rows], dtype=np.float64)
    art = np.asarray([r.artifact_rate for r in rows], dtype=np.float64)
    snr = np.asarray([r.snr_proxy for r in rows], dtype=np.float64)

    alpha_n = _minmax(alpha)
    hf_n = _minmax(hf)
    art_n = _minmax(art)
    snr_n = _minmax(snr)

    score = 0.35 * snr_n + 0.25 * alpha_n + 0.20 * (1.0 - hf_n) + 0.20 * (1.0 - art_n)

    out: list[SubjectQuality] = []
    for idx, row in enumerate(rows):
        copy = SubjectQuality(**asdict(row))
        copy.quality_score = float(score[idx])
        out.append(copy)
    return out


def _print_ranked_table(rows: list[SubjectQuality]) -> None:
    print("\nPer-subject signal quality ranking (higher is better)")
    print(
        "subject | score  | trials | artifact_rate | alpha_ratio | hf_ratio | snr_proxy"
    )
    print("-" * 79)
    for row in sorted(rows, key=lambda r: r.quality_score, reverse=True):
        print(
            f"{row.subject_id:>7d} | "
            f"{row.quality_score:>0.4f} | "
            f"{row.n_trials:>6d} | "
            f"{row.artifact_rate:>0.4f}       | "
            f"{row.alpha_ratio:>0.4f}     | "
            f"{row.hf_ratio:>0.4f}   | "
            f"{row.snr_proxy:>0.4f}"
        )


def _write_json(
    path: Path, rows: list[SubjectQuality], config: dict[str, object]
) -> None:
    payload = {
        "config": config,
        "subjects": [asdict(r) for r in rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[SubjectQuality]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def run(args: argparse.Namespace) -> int:
    if args.data_path:
        os.environ["MNE_DATA"] = str(Path(args.data_path).expanduser().resolve())

    options = MoabbLoadOptions(
        class_policy=args.class_policy,
        subjects=args.subjects if args.subjects else None,
        max_subjects=args.max_subjects,
        resample=args.resample,
        tmin=args.tmin,
        tmax=args.tmax,
        fmin=args.fmin,
        fmax=args.fmax,
        include_metadata=False,
        show_progress=True,
    )

    x, _, subject_id, subjects = load_moabb_motor_imagery_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        options=options,
    )

    rows: list[SubjectQuality] = []
    for sid in subjects:
        mask = subject_id == int(sid)
        x_subject = x[mask]
        if x_subject.size == 0:
            continue
        rows.append(
            _subject_quality_metrics(
                x_subject=x_subject,
                sfreq=float(args.resample),
                subject_id=int(sid),
            )
        )

    if not rows:
        print(
            "No subject rows produced; check dataset/subject filters.", file=sys.stderr
        )
        return 1

    rows = _attach_quality_scores(rows)
    _print_ranked_table(rows)

    config = {
        "dataset": args.dataset,
        "class_policy": args.class_policy,
        "resample": args.resample,
        "tmin": args.tmin,
        "tmax": args.tmax,
        "fmin": args.fmin,
        "fmax": args.fmax,
        "subjects": args.subjects,
        "max_subjects": args.max_subjects,
    }

    if args.json_out:
        json_path = Path(args.json_out).expanduser()
        _write_json(json_path, rows, config=config)
        print(f"\nSaved JSON report: {json_path}")

    if args.csv_out:
        csv_path = Path(args.csv_out).expanduser()
        _write_csv(csv_path, rows)
        print(f"Saved CSV report: {csv_path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare per-subject EEG signal quality metrics for MOABB BCI datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bnci2014_001",
        choices=["bnci2014_001", "physionetmi", "cho2017", "lee2019_mi"],
    )
    parser.add_argument(
        "--data_path", type=str, default=os.environ.get("MNE_DATA", None)
    )
    parser.add_argument(
        "--class_policy",
        type=str,
        default="left_right",
        choices=["left_right", "all_mi"],
    )
    parser.add_argument("--subjects", nargs="*", type=int, default=None)
    parser.add_argument("--max_subjects", type=int, default=None)

    parser.add_argument("--resample", type=int, default=128)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument("--fmin", type=float, default=4.0)
    parser.add_argument("--fmax", type=float, default=40.0)

    parser.add_argument("--json_out", type=str, default="")
    parser.add_argument("--csv_out", type=str, default="")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
