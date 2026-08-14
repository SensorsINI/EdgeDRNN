"""Versioned, native PyTorch checkpoint I/O."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

CHECKPOINT_FORMAT_VERSION = 1


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    val_loss: float,
    config: Mapping[str, Any],
) -> None:
    """Atomically save a checkpoint containing only safe, native objects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": dict(config),
    }
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Load a native checkpoint without permitting arbitrary pickle objects."""
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {path} does not contain a dictionary")
    return checkpoint


def model_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    """Accept versioned checkpoints and plain state-dict containers."""
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if checkpoint and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        return checkpoint
    raise KeyError("Checkpoint does not contain model weights")


def load_model_state(
    model: nn.Module,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> nn.Module:
    checkpoint = load_checkpoint(path, map_location=map_location)
    model.load_state_dict(model_state_dict(checkpoint), strict=strict)
    return model
