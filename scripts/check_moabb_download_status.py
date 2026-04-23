#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class DatasetCheckResult:
    dataset: str
    code: str
    total_subjects: int
    checked_subjects: int
    loaded_subjects: int
    failed_subjects: int
    failed_subject_ids: list[int]
    failed_reasons: dict[int, str]


_COMMON_CHANNELS = [
    "FC3", "FC1", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
]


def _build_dataset(name: str) -> tuple[Any, str]:
    from moabb.datasets import BNCI2014_001, Cho2017, Lee2019_MI, PhysionetMI
    from moabb.paradigms import LeftRightImagery, MotorImagery

    key = name.lower()
    if key in {"bnci2014_001", "bnci", "bci"}:
        dataset = BNCI2014_001()
        paradigm = LeftRightImagery(
            fmin=4.0, fmax=40.0, tmin=0.0, tmax=4.0,
            resample=128, channels=_COMMON_CHANNELS,
        )
        return (dataset, paradigm), "bnci2014_001"
    if key in {"physionetmi", "physionet"}:
        dataset = PhysionetMI(imagined=True, executed=False)
        paradigm = LeftRightImagery(
            fmin=4.0, fmax=40.0, tmin=0.0, tmax=4.0,
            resample=128, channels=_COMMON_CHANNELS,
        )
        return (dataset, paradigm), "physionetmi"
    if key in {"cho2017", "cho"}:
        dataset = Cho2017()
        paradigm = LeftRightImagery(
            fmin=4.0, fmax=40.0, tmin=0.0, tmax=4.0,
            resample=128, channels=_COMMON_CHANNELS,
        )
        return (dataset, paradigm), "cho2017"
    if key in {"lee2019_mi", "lee2019", "lee"}:
        dataset = Lee2019_MI()
        paradigm = LeftRightImagery(
            fmin=4.0, fmax=40.0, tmin=0.0, tmax=4.0,
            resample=128, channels=_COMMON_CHANNELS,
        )
        return (dataset, paradigm), "lee2019_mi"
    raise ValueError(
        f"Unsupported dataset: {name}. Use one of: bnci2014_001, physionetmi, cho2017, lee2019_mi"
    )


def run(args: argparse.Namespace) -> int:
    data_path = str(Path(args.data_path).expanduser().resolve())
    os.environ["MNE_DATA"] = data_path

    try:
        import mne
    except ImportError:
        print("ERROR: MNE-Python not installed.", file=sys.stderr)
        return 1

    try:
        import moabb  # noqa: F401
    except ImportError:
        print("ERROR: MOABB not installed.", file=sys.stderr)
        return 1

    mne.set_log_level("ERROR")
    logging.getLogger("moabb").setLevel(logging.ERROR)
    logging.getLogger("mne").setLevel(logging.ERROR)

    all_results: list[DatasetCheckResult] = []

    for dataset_name in args.datasets:
        (dataset, paradigm), canonical_name = _build_dataset(dataset_name)
        all_subjects = [int(s) for s in dataset.subject_list]
        subjects = all_subjects
        if args.max_subjects is not None:
            if args.max_subjects <= 0:
                raise ValueError("--max-subjects must be > 0")
            subjects = all_subjects[: args.max_subjects]

        loaded_subjects: list[int] = []
        failed_subjects: list[int] = []
        failed_reasons: dict[int, str] = {}

        for sid in subjects:
            try:
                result = paradigm.get_data(dataset=dataset, subjects=[int(sid)])
                if hasattr(result, "keys"):
                    keys = list(result.keys())
                    if len(keys) == 0:
                        failed_subjects.append(int(sid))
                        failed_reasons[int(sid)] = "empty_result"
                    else:
                        first_val = result[keys[0]]
                        if hasattr(first_val, "n_times"):
                            loaded_subjects.append(int(sid))
                        else:
                            failed_subjects.append(int(sid))
                            failed_reasons[int(sid)] = f"unhandled({type(first_val).__name__})"
                elif hasattr(result, "__iter__"):
                    items = list(result)
                    first = items[0] if items else None
                    if first is not None and hasattr(first, "n_times"):
                        loaded_subjects.append(int(sid))
                    else:
                        failed_subjects.append(int(sid))
                        failed_reasons[int(sid)] = "unhandled_generator"
                else:
                    failed_subjects.append(int(sid))
                    failed_reasons[int(sid)] = f"unhandled_return({type(result).__name__})"
            except Exception as e:
                failed_subjects.append(int(sid))
                failed_reasons[int(sid)] = str(e)[:100]

        result = DatasetCheckResult(
            dataset=canonical_name,
            code=str(getattr(dataset, "code", canonical_name)),
            total_subjects=len(all_subjects),
            checked_subjects=len(subjects),
            loaded_subjects=len(loaded_subjects),
            failed_subjects=len(failed_subjects),
            failed_subject_ids=failed_subjects,
            failed_reasons=failed_reasons,
        )
        all_results.append(result)

        status = "OK" if len(failed_subjects) == 0 else "PARTIAL" if loaded_subjects else "FAIL"
        print(
            f"[{status}] {canonical_name}: "
            f"{result.loaded_subjects}/{result.checked_subjects} loaded "
            f"(total={result.total_subjects})"
        )
        if failed_subjects:
            tail = f", +{len(failed_subjects)-20} more" if len(failed_subjects) > 20 else ""
            print(f"       failed ({len(failed_subjects)}): {failed_subjects[:20]}{tail}")
            if args.verbose:
                for sid in failed_subjects[:10]:
                    print(f"         s{sid}: {failed_reasons.get(sid, 'unknown')}")

    if args.json_out:
        out_path = Path(args.json_out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(r) for r in all_results]
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report: {out_path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check how many MOABB subjects load successfully per dataset"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["physionetmi", "cho2017", "lee2019_mi", "bnci2014_001"],
        choices=["physionetmi", "cho2017", "lee2019_mi", "bnci2014_001"],
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.environ.get("MNE_DATA", str(Path.home() / "mne_data")),
    )
    parser.add_argument("--max_subjects", type=int, default=None)
    parser.add_argument("--json_out", type=str, default="")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))