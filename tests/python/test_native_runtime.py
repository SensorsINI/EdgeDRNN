"""Unit tests for the native PyTorch runtime infrastructure."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from edgedrnn.checkpoint import load_checkpoint, load_model_state, save_checkpoint
from edgedrnn.config import ProjectConfig
from torch import nn


class NativeRuntimeTest(unittest.TestCase):
    def test_run_through_selects_remaining_stages(self) -> None:
        config = ProjectConfig(
            ["--step", "retrain", "--run-through", "--num-workers", "0"]
        )
        self.assertEqual(config.selected_steps(), ("retrain", "export"))

    def test_native_checkpoint_round_trip(self) -> None:
        source = nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "best.pt"
            save_checkpoint(
                checkpoint_path,
                model=source,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=3,
                val_loss=1.25,
                config={"rnn_type": "GRU"},
            )

            checkpoint = load_checkpoint(checkpoint_path)
            self.assertEqual(checkpoint["format_version"], 1)
            self.assertEqual(checkpoint["epoch"], 3)
            self.assertEqual(checkpoint["config"]["rnn_type"], "GRU")

            restored = nn.Linear(3, 2)
            load_model_state(restored, checkpoint_path)
            for expected, actual in zip(source.parameters(), restored.parameters()):
                torch.testing.assert_close(expected, actual)


if __name__ == "__main__":
    unittest.main()
