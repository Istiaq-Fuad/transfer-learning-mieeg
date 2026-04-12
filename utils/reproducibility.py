from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed_everywhere(seed: int, deterministic: bool = True) -> None:
    """Set RNG seeds across python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_torch_generator(seed: int) -> torch.Generator:
    """Build a torch generator with a deterministic seed."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """Seed dataloader worker subprocess RNGs from torch initial seed."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _stable_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: dict[str, Any]) -> str:
    payload = _stable_json(config).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


@dataclass
class ExperimentMetadata:
    experiment_id: str
    config_hash: str
    seed: int
    deterministic: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "config_hash": self.config_hash,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "notes": self.notes,
        }


def create_experiment_metadata(
    protocol: str,
    dataset: str,
    config: dict[str, Any],
    seed: int,
    deterministic: bool = True,
    notes: str = "",
) -> ExperimentMetadata:
    cfg_hash = config_hash(config)
    exp_id = f"{protocol}_{dataset}_{cfg_hash}"
    return ExperimentMetadata(
        experiment_id=exp_id,
        config_hash=cfg_hash,
        seed=seed,
        deterministic=deterministic,
        notes=notes,
    )


def save_experiment_metadata(path: Path, metadata: ExperimentMetadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, indent=2)
