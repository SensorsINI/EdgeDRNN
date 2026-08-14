"""Small, explicit native PyTorch training engine."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from edgedrnn import utils
from edgedrnn.checkpoint import load_model_state, save_checkpoint
from edgedrnn.config import ProjectConfig


def resolve_device(requested: str) -> torch.device:
    normalized = "cuda" if requested == "gpu" else requested
    if normalized == "auto":
        if torch.cuda.is_available():
            normalized = "cuda"
        elif torch.backends.mps.is_available():
            normalized = "mps"
        else:
            normalized = "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if normalized == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(normalized)


def _move_batch(batch, device: torch.device):
    non_blocking = device.type == "cuda"
    features, labels = batch
    return (
        features.to(device, non_blocking=non_blocking),
        labels.to(device, non_blocking=non_blocking),
    )


def _run_evaluation(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    weighted_loss = 0.0
    sample_count = 0
    with torch.inference_mode():
        for batch in loader:
            features, labels = _move_batch(batch, device)
            loss = model.compute_loss((features, labels))
            batch_size = features.shape[0]
            weighted_loss += loss.item() * batch_size
            sample_count += batch_size
    if sample_count == 0:
        raise RuntimeError("Evaluation DataLoader is empty")
    return weighted_loss / sample_count


def _quantize_weights(model: nn.Module, config: ProjectConfig) -> None:
    if not config.qw:
        return
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(
                utils.quantize_tensor(parameter, config.wqi, config.wqf, config.qw)
            )


def _write_metrics_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow(("epoch", "learning_rate", "train_loss", "val_loss"))


def _append_metrics(
    path: Path,
    *,
    epoch: int,
    learning_rate: float,
    train_loss: float,
    val_loss: float,
) -> None:
    with path.open("a", newline="", encoding="utf-8") as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow((epoch, learning_rate, train_loss, val_loss))


def train(config: ProjectConfig) -> dict[str, float | str]:
    """Train the configured stage, save best/last checkpoints, and test the best model."""
    config.reproducible()
    dataloaders = config.prepare_dataloader()
    model = config.prepare_model()
    if config.experiment_key is None:
        raise RuntimeError("Experiment key was not initialized")

    device = resolve_device(config.device)
    model.to(device)
    optimizer = AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    checkpoint_dir = config.checkpoint_dir(config.step, config.experiment_key)
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    metrics_path = config.log_dir(config.step, config.experiment_key) / "metrics.csv"
    _write_metrics_header(metrics_path)

    print(f"Training on {device} with PyTorch {torch.__version__}")
    best_val_loss = math.inf
    for epoch in range(config.max_epochs):
        model.train()
        weighted_train_loss = 0.0
        sample_count = 0
        for batch in dataloaders.train_loader:
            features, labels = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss((features, labels))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_val)
            optimizer.step()

            batch_size = features.shape[0]
            weighted_train_loss += loss.detach().item() * batch_size
            sample_count += batch_size

        if sample_count == 0:
            raise RuntimeError("Training DataLoader is empty")
        train_loss = weighted_train_loss / sample_count

        # Preserve the original quantization-aware schedule: project weights to the
        # configured fixed-point grid before validation and the next epoch.
        _quantize_weights(model, config)
        val_loss = _run_evaluation(model, dataloaders.dev_loader, device)
        scheduler.step(val_loss)
        learning_rate = optimizer.param_groups[0]["lr"]

        checkpoint_config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in config.hparams.items()
        }
        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            val_loss=val_loss,
            config=checkpoint_config,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_loss=val_loss,
                config=checkpoint_config,
            )

        _append_metrics(
            metrics_path,
            epoch=epoch,
            learning_rate=learning_rate,
            train_loss=train_loss,
            val_loss=val_loss,
        )
        print(
            f"Epoch {epoch + 1:02d}/{config.max_epochs:02d} | "
            f"lr={learning_rate:.6g} | train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f}"
        )

    load_model_state(model, best_path, map_location=device)
    test_loss = _run_evaluation(model, dataloaders.test_loader, device)
    print(f"test_loss={test_loss:.6f} | best_checkpoint={best_path}")
    return {
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "checkpoint": str(best_path),
    }
