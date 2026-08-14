"""Command-line configuration and workflow setup for EdgeDRNN."""

from __future__ import annotations

import argparse
import importlib
import os
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


class ProjectConfig:
    """Runtime configuration shared by training and hardware export."""

    STEPS = ("pretrain", "retrain", "export")

    def __init__(self, argv: Sequence[str] | None = None):
        torch.set_float32_matmul_precision("high")
        torch.set_printoptions(precision=8)

        self.repo_root = Path(__file__).resolve().parents[2]
        self.python_root = self.repo_root / "python"
        self.num_cpu_threads = os.cpu_count() or 1

        parser = argparse.ArgumentParser(
            description="Train, quantize, and export an EdgeDRNN model with native PyTorch."
        )
        basic = parser.add_argument_group("Basic setup")
        basic.add_argument("--dataset-name", "--dataset_name", default="cps")
        basic.add_argument("--step", choices=self.STEPS, default="pretrain")
        basic.add_argument(
            "--run-through",
            "--run_through",
            nargs="?",
            const=True,
            default=False,
            type=_parse_bool,
            help="Run the selected step and every following step.",
        )
        basic.add_argument(
            "--device",
            "--accelerator",
            dest="device",
            choices=("auto", "cpu", "cuda", "gpu", "mps"),
            default="auto",
            help="Execution device. The legacy value 'gpu' maps to CUDA.",
        )
        basic.add_argument(
            "--num-workers",
            type=int,
            default=min(8, max(0, self.num_cpu_threads // 4)),
            help="DataLoader worker processes.",
        )
        basic.add_argument(
            "--model-path",
            "--model_path",
            default="",
            help="Checkpoint file or directory to load instead of the default best.pt.",
        )
        basic.add_argument(
            "--output-dir",
            default=str(self.repo_root / "artifacts"),
            help="Directory for checkpoints and training logs.",
        )
        basic.add_argument(
            "--stimuli-dir",
            default=str(self.repo_root / "verification" / "rtl" / "stimuli"),
            help="Directory for generated RTL stimuli and lookup tables.",
        )
        basic.add_argument(
            "--sdk-dir",
            default=str(self.python_root / "sdk"),
            help="Directory for generated C and MATLAB artifacts.",
        )

        features = parser.add_argument_group("Dataset processing")
        features.add_argument(
            "--normalize-features", "--normalize_features", default=1, type=int
        )
        features.add_argument("--batch-first", "--batch_first", default=1, type=int)

        training = parser.add_argument_group("Training hyperparameters")
        training.add_argument("--seed", default=2, type=int)
        training.add_argument(
            "--epochs-pretrain", "--epochs_pretrain", default=10, type=int
        )
        training.add_argument(
            "--epochs-retrain", "--epochs_retrain", default=2, type=int
        )
        training.add_argument("--batch-size", "--batch_size", default=128, type=int)
        training.add_argument(
            "--batch-size-test", "--batch_size_test", default=256, type=int
        )
        training.add_argument("--lr", default=3e-4, type=float)
        training.add_argument(
            "--weight-decay", "--weight_decay", default=0.01, type=float
        )
        training.add_argument(
            "--grad-clip-val", "--grad_clip_val", default=200.0, type=float
        )
        training.add_argument("--ctxt-size", "--ctxt_size", default=50, type=int)
        training.add_argument("--pred-size", "--pred_size", default=0, type=int)
        training.add_argument(
            "--deterministic",
            nargs="?",
            const=True,
            default=True,
            type=_parse_bool,
            help="Request deterministic PyTorch algorithms when available.",
        )

        model = parser.add_argument_group("Model hyperparameters")
        model.add_argument("--rnn-layers", "--rnn_layers", default=2, type=int)
        model.add_argument("--rnn-size", "--rnn_size", default=16, type=int)
        model.add_argument("--rnn-type-pretrain", "--rnn_type_pretrain", default="GRU")
        model.add_argument(
            "--rnn-type-retrain", "--rnn_type_retrain", default="DeltaGRU"
        )
        model.add_argument("--rnn-dropout", "--rnn_dropout", default=0.1, type=float)

        quantization = parser.add_argument_group("Quantization hyperparameters")
        quantization.add_argument("--qa", default=1, type=int)
        quantization.add_argument("--qw", default=1, type=int)
        quantization.add_argument("--qaf", default=1, type=int)
        quantization.add_argument("--qc", default=1, type=int)
        quantization.add_argument("--aqi", default=8, type=int)
        quantization.add_argument("--aqf", default=8, type=int)
        quantization.add_argument("--wqi", default=1, type=int)
        quantization.add_argument("--wqf", default=7, type=int)
        quantization.add_argument("--afqi", default=2, type=int)
        quantization.add_argument("--afqf", default=4, type=int)
        quantization.add_argument("--cqi", default=8, type=int)
        quantization.add_argument("--cqf", default=8, type=int)

        delta = parser.add_argument_group("Delta network hyperparameters")
        delta.add_argument("--thx", default=0.0, type=float)
        delta.add_argument("--thh", default=0.0, type=float)

        hardware = parser.add_argument_group("EdgeDRNN hardware")
        hardware.add_argument("--num-pe", "--num_pe", default=8, type=int)
        hardware.add_argument("--stim-head", "--stim_head", default=1000, type=int)
        hardware.add_argument("--stim-len", "--stim_len", default=1000, type=int)

        self.parser = parser
        self.args = parser.parse_args(argv)
        self.args.batch_first = bool(self.args.batch_first)
        self.args.normalize_features = bool(self.args.normalize_features)
        self.args.run_through = bool(self.args.run_through)
        self.args.output_dir = str(Path(self.args.output_dir).resolve())
        self.args.stimuli_dir = str(Path(self.args.stimuli_dir).resolve())
        self.args.sdk_dir = str(Path(self.args.sdk_dir).resolve())
        self.args.accelerator = self.args.device  # Compatibility for dataset plugins.

        self.hparams = vars(self.args).copy()
        for key, value in self.hparams.items():
            setattr(self, key, value)

        if self.rnn_size % self.num_pe:
            parser.error("--rnn-size must be a multiple of --num-pe")
        if self.num_workers < 0:
            parser.error("--num-workers must be non-negative")

        self.args_to_abb = {
            "seed": "S",
            "inp_size": "I",
            "rnn_size": "H",
            "rnn_type": "T",
            "rnn_layers": "L",
            "num_classes": "C",
            "ctxt_size": "CT",
            "pred_size": "PD",
            "qa": "QA",
            "aqi": "AQI",
            "aqf": "AQF",
            "qw": "QW",
            "wqi": "WQI",
            "wqf": "WQF",
        }
        self.experiment_key: str | None = None
        self.step_idx = self.STEPS.index(self.step)

    def update_args(self, args: argparse.Namespace) -> None:
        self.args = args
        for key, value in vars(args).items():
            setattr(self, key, value)
            self.hparams[key] = value

    def update_attr(self, key: str, value) -> None:
        setattr(self, key, value)
        setattr(self.args, key, value)
        self.hparams[key] = value

    def selected_steps(self) -> tuple[str, ...]:
        if self.run_through:
            return self.STEPS[self.step_idx :]
        return (self.step,)

    def set_step(self, step: str) -> None:
        self.update_attr("step", step)
        self.step_idx = self.STEPS.index(step)

    def gen_experiment_key(self) -> str:
        parts: list[str] = []
        for key, abbreviation in self.args_to_abb.items():
            parts.extend((abbreviation, str(self.hparams[key])))
        self.experiment_key = "_".join(parts)
        return self.experiment_key

    def prepare_dataloader(self):
        try:
            module = importlib.import_module(
                f"edgedrnn.datasets.{self.dataset_name}.dataloader"
            )
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                f"Dataset {self.dataset_name!r} is not available"
            ) from error
        return module.DataLoader(self)

    def prepare_model(self):
        from edgedrnn.model import EdgeDRNNModel

        if self.step == "pretrain":
            self.update_attr("rnn_type", self.rnn_type_pretrain)
            self.update_attr("max_epochs", self.epochs_pretrain)
            self.gen_experiment_key()
            return EdgeDRNNModel(self.args)

        if self.step == "retrain":
            self.update_attr("rnn_type", self.rnn_type_pretrain)
            self.gen_experiment_key()
            pretrained_path = self.resolve_checkpoint("pretrain", self.experiment_key)
            print(f"Loading pretrained model: {pretrained_path}")
            self.update_attr("rnn_type", self.rnn_type_retrain)
            self.update_attr("max_epochs", self.epochs_retrain)
            self.gen_experiment_key()
            return EdgeDRNNModel(self.args).load_gru_weight(pretrained_path)

        if self.step == "export":
            self.update_attr("rnn_type", self.rnn_type_retrain)
            self.gen_experiment_key()
            checkpoint_path = self.resolve_checkpoint("retrain", self.experiment_key)
            print(f"Loading retrained model: {checkpoint_path}")
            return EdgeDRNNModel(self.args).load_deltagru_weight(checkpoint_path)

        raise ValueError(f"Unsupported step: {self.step}")

    def checkpoint_dir(self, step: str, experiment_key: str) -> Path:
        return Path(self.output_dir) / "checkpoints" / step / experiment_key

    def log_dir(self, step: str, experiment_key: str) -> Path:
        return Path(self.output_dir) / "logs" / step / experiment_key

    def resolve_checkpoint(self, step: str, experiment_key: str) -> Path:
        if self.model_path:
            candidate = Path(self.model_path).expanduser().resolve()
        else:
            candidate = self.checkpoint_dir(step, experiment_key) / "best.pt"

        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            preferred = candidate / "best.pt"
            if preferred.is_file():
                return preferred
            checkpoints = sorted((*candidate.glob("*.pt"), *candidate.glob("*.ckpt")))
            if checkpoints:
                return checkpoints[-1]
        raise FileNotFoundError(f"No checkpoint found at {candidate}")

    def reproducible(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = self.deterministic
        torch.use_deterministic_algorithms(self.deterministic, warn_only=True)


# Backward-compatible type name for dataset plugins and downstream scripts.
Project = ProjectConfig
