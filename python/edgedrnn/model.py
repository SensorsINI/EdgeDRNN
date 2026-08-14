"""EdgeDRNN neural-network definitions built directly on ``torch.nn``."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import torch
from torch import nn

from edgedrnn import utils
from edgedrnn.checkpoint import load_checkpoint, model_state_dict
from edgedrnn.layers.deltagru import DeltaGRU


class EdgeDRNNModel(nn.Module):
    """GRU/DeltaGRU regression model used by training and RTL export."""

    def __init__(self, args: Namespace):
        super().__init__()
        for key, value in vars(args).items():
            setattr(self, key, value)

        self.accqi = args.aqi + args.wqi
        self.accqf = args.aqf + args.wqf
        self.debug = 0
        self.statistics: dict[str, float | int] = {}
        self.abs_delta_hid = 0
        self.abs_std_delta_hid = 0
        self.all_layer_dx: list[torch.Tensor] = []
        self.all_layer_dh: list[torch.Tensor] = []
        self.list_debug: list[torch.Tensor] = []
        self.list_log: list[dict[str, torch.Tensor]] = []

        self.loss = nn.MSELoss()
        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=self.inp_size,
                hidden_size=self.rnn_size,
                num_layers=self.rnn_layers,
                bias=True,
                bidirectional=False,
                batch_first=self.batch_first,
                dropout=self.rnn_dropout if self.rnn_layers > 1 else 0.0,
            )
        elif self.rnn_type == "DeltaGRU":
            self.rnn = DeltaGRU(
                input_size=self.inp_size,
                hidden_size=self.rnn_size,
                num_layers=self.rnn_layers,
                batch_first=self.batch_first,
                thx=self.thx,
                thh=self.thh,
                qa=self.qa,
                aqi=self.aqi,
                aqf=self.aqf,
                nqi=self.afqi,
                nqf=self.afqf,
                debug=0,
            )
        else:
            raise ValueError(f"Unsupported recurrent layer: {self.rnn_type}")

        self.cl = nn.Linear(self.rnn_size, self.num_classes, bias=True)
        self.reset_parameters()

    def set_qa(self, enabled: int) -> None:
        self.qa = enabled
        if "Delta" in self.rnn_type:
            self.rnn.set_qa(enabled)

    def set_eval_sp(self, enabled: int) -> None:
        self.eval_sp = enabled
        if "Delta" in self.rnn_type:
            self.rnn.set_eval_sparsity(enabled)

    def reset_parameters(self) -> None:
        """Initialize each recurrent gate independently and the output projection."""
        for name, parameter in self.named_parameters():
            if "rnn" in name and "weight" in name:
                for gate in parameter.chunk(3, dim=0):
                    nn.init.orthogonal_(gate)
            elif name == "cl.weight":
                nn.init.xavier_uniform_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.list_log = []
        classification_log: dict[str, torch.Tensor] = {}
        quantize_classification = bool(self.qc and not self.training)

        if self.rnn_type in {"GRU", "LSTM"}:
            self.rnn.flatten_parameters()

        if "Delta" in self.rnn_type:
            recurrent_output, _, _ = self.rnn(inputs)
        else:
            recurrent_output, _ = self.rnn(inputs)
        classification_output = self.cl(recurrent_output)

        accumulated_output = utils.quantize_tensor(
            classification_output, self.accqi, self.accqf, self.qc
        )
        quantized_output = utils.quantize_tensor(
            classification_output, self.cqi, self.cqf, quantize_classification
        )

        if not self.training and self.rnn_type == "DeltaGRU":
            classification_log["cl_inp"] = recurrent_output
            classification_log["cl_out"] = classification_output
            classification_log["cl_out_acc"] = accumulated_output
            self.list_log = self.rnn.list_log
            self.list_log.append(classification_log)

        return quantized_output, recurrent_output

    def compute_loss(self, batch) -> torch.Tensor:
        features, labels = batch
        outputs, _ = self(features)
        return self.loss(outputs, labels)

    def load_gru_weight(self, model_path: str | Path) -> EdgeDRNNModel:
        checkpoint = load_checkpoint(model_path, map_location="cpu")
        self.load_state_dict(model_state_dict(checkpoint), strict=True)
        with torch.no_grad():
            self.rnn.process_biases()
        return self

    def load_deltagru_weight(self, model_path: str | Path) -> EdgeDRNNModel:
        checkpoint = load_checkpoint(model_path, map_location="cpu")
        self.load_state_dict(model_state_dict(checkpoint), strict=True)
        return self

    def get_sparsity(self) -> dict[str, float | int]:
        for name, module in self._modules.items():
            self.statistics[f"SP_W_{name.upper()}"] = utils.get_layer_sparsity(module)
        self.statistics.update(self.rnn.get_temporal_sparsity())
        return self.statistics

    def get_model_size(self) -> int:
        model_size = sum(parameter.numel() for parameter in self.parameters())
        self.statistics["NUM_PARAMS"] = model_size
        return model_size


# Preserve the original import name for external scripts.
Model = EdgeDRNNModel
