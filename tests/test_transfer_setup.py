from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data.loader import (
    DataLoaderOptions,
    create_loso_domain_adaptation_dataloaders,
)
from models.model import EEGModel
from training.loso import _load_init_checkpoint


def _loader_sample_ids(loader: torch.utils.data.DataLoader) -> set[int]:
    sample_ids: set[int] = set()
    for xb, _, _ in loader:
        sample_ids.update(int(v.item()) for v in xb[:, 0, 0])
    return sample_ids


class TransferSetupTests(unittest.TestCase):
    def test_loso_da_target_adaptation_and_test_are_disjoint(self) -> None:
        x = np.arange(24, dtype=np.float32).reshape(24, 1, 1)
        y = np.asarray([0, 1] * 12, dtype=np.int64)
        subject_id = np.asarray([1] * 12 + [2] * 12, dtype=np.int64)

        source_loader, target_loader, test_loader = (
            create_loso_domain_adaptation_dataloaders(
                x=x,
                y=y,
                subject_id=subject_id,
                target_subject=2,
                options=DataLoaderOptions(
                    batch_size=4,
                    target_test_size=0.25,
                    random_state=7,
                    apply_euclidean_align=False,
                    seed=7,
                    deterministic=True,
                ),
            )
        )

        source_ids = _loader_sample_ids(source_loader)
        target_ids = _loader_sample_ids(target_loader)
        test_ids = _loader_sample_ids(test_loader)

        self.assertTrue(target_ids)
        self.assertTrue(test_ids)
        self.assertTrue(target_ids.isdisjoint(test_ids))
        self.assertTrue(source_ids.isdisjoint(test_ids))

        test_labels: set[int] = set()
        for _, yb, _ in test_loader:
            test_labels.update(int(v.item()) for v in yb)
        self.assertEqual(test_labels, {0, 1})

    def test_supervised_checkpoint_loads_task_head_and_skips_domain_heads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir()
            (run_dir / "config.json").write_text(
                '{"pretrain_mode": "supervised"}',
                encoding="utf-8",
            )

            source = EEGModel(num_channels=2, num_classes=2, num_subjects=5)
            with torch.no_grad():
                source.task_head.classifier.weight.fill_(7.0)
                source.task_head.classifier.bias.fill_(3.0)
                source.domain_head.classifier.weight.fill_(11.0)
            ckpt_path = ckpt_dir / "pretrain_best.pt"
            torch.save(source.state_dict(), ckpt_path)

            target = EEGModel(num_channels=2, num_classes=2, num_subjects=5)
            _load_init_checkpoint(
                target,
                str(ckpt_path),
                torch.device("cpu"),
                logging.getLogger("test_supervised_checkpoint"),
            )

            self.assertTrue(
                torch.allclose(
                    target.task_head.classifier.weight,
                    torch.full_like(target.task_head.classifier.weight, 7.0),
                )
            )
            self.assertFalse(
                torch.allclose(
                    target.domain_head.classifier.weight,
                    torch.full_like(target.domain_head.classifier.weight, 11.0),
                )
            )

    def test_ssl_checkpoint_auto_skips_task_head(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir()
            (run_dir / "config.json").write_text(
                '{"pretrain_mode": "ssl"}',
                encoding="utf-8",
            )

            source = EEGModel(num_channels=2, num_classes=2, num_subjects=5)
            with torch.no_grad():
                source.task_head.classifier.weight.fill_(7.0)
                source.task_head.classifier.bias.fill_(3.0)
            ckpt_path = ckpt_dir / "pretrain_best.pt"
            torch.save(source.state_dict(), ckpt_path)

            target = EEGModel(num_channels=2, num_classes=2, num_subjects=5)
            initial_weight = target.task_head.classifier.weight.detach().clone()
            initial_bias = target.task_head.classifier.bias.detach().clone()

            _load_init_checkpoint(
                target,
                str(ckpt_path),
                torch.device("cpu"),
                logging.getLogger("test_ssl_checkpoint"),
            )

            self.assertTrue(
                torch.allclose(target.task_head.classifier.weight, initial_weight)
            )
            self.assertTrue(torch.allclose(target.task_head.classifier.bias, initial_bias))


if __name__ == "__main__":
    unittest.main()
