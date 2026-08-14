"""Export trained DeltaGRU models to the EdgeDRNN hardware formats."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy import io

from edgedrnn import utils
from edgedrnn.config import ProjectConfig


def round_up_to_multiple(value: int, multiple: int) -> int:
    return int(multiple * np.ceil(value / multiple))


class EdgeDRNNExporter:
    """Generate SDK assets, simulator stimuli, golden data, and LUTs."""

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.stimuli_dir = Path(config.stimuli_dir)
        self.sdk_dir = Path(config.sdk_dir)
        self.stimuli_dir.mkdir(parents=True, exist_ok=True)
        self.sdk_dir.mkdir(parents=True, exist_ok=True)

        self.dataloaders = config.prepare_dataloader()
        self.model = config.prepare_model().cpu().eval()

        input_size_hardware = round_up_to_multiple(config.inp_size, config.num_pe)
        config.update_attr("inp_size_hw", input_size_hardware)
        print("EdgeDRNN hardware configuration")
        print(f"  Processing elements: {config.num_pe}")
        print(f"  RNN input size (software): {config.inp_size}")
        print(f"  RNN input size (hardware): {config.inp_size_hw}")

        self.edgedrnn_params: np.ndarray | None = None
        self.cl_weight: np.ndarray | None = None
        self.cl_bias: np.ndarray | None = None

    def export(self) -> None:
        self.collect_params()
        self.collect_test_data()
        self.generate_luts()
        self.write_stimuli_manifest()
        print(f"EdgeDRNN export completed: {self.stimuli_dir}")

    def collect_params(self) -> None:
        print("Collecting network parameters")
        recurrent_weights: list[np.ndarray] = []
        recurrent_biases: list[np.ndarray] = []
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                parameter.copy_(
                    utils.quantize_tensor(
                        parameter, self.config.wqi, self.config.wqf, self.config.qw
                    )
                )
                value = parameter.detach().cpu().numpy()
                if "rnn" in name and "weight" in name:
                    recurrent_weights.append(value)
                elif "rnn" in name and "bias_ih" in name:
                    recurrent_biases.append(value.reshape(-1, 1))
                elif "cl" in name and "weight" in name:
                    self.cl_weight = value
                elif "cl" in name and "bias" in name:
                    self.cl_bias = value.reshape(-1, 1)

        if self.cl_weight is None or self.cl_bias is None:
            raise RuntimeError("Classification parameters were not found")

        pad_length = self.config.inp_size_hw - self.config.inp_size
        input_padding = np.zeros((3 * self.config.rnn_size, pad_length))
        hardware_parameters: list[np.ndarray] = []
        for layer_index in range(self.config.rnn_layers):
            hardware_parameters.append(recurrent_biases[layer_index])
            hardware_parameters.append(recurrent_weights[2 * layer_index])
            if layer_index == 0:
                hardware_parameters.append(input_padding)
            hardware_parameters.append(recurrent_weights[2 * layer_index + 1])
        self.edgedrnn_params = np.hstack(hardware_parameters)

        sdk_parameters = {
            "rnn_param": (
                self.edgedrnn_params,
                1,
                self.config.wqi,
                self.config.wqf,
                "param",
                16,
            ),
            "cl_bias": (self.cl_bias, 1, self.config.wqi, self.config.wqf, "param", 16),
            "cl_weight": (
                self.cl_weight,
                1,
                self.config.wqi,
                self.config.wqf,
                "param",
                16,
            ),
            "rnn_layers": (self.config.rnn_layers, 0, 16, 0, "const", 1),
            "inp_size": (self.config.inp_size_hw, 0, 16, 0, "const", 1),
            "rnn_size": (self.config.rnn_size, 0, 16, 0, "const", 1),
            "thx": (self.config.thx, 1, self.config.aqi, self.config.aqf, "const", 1),
            "thh": (self.config.thh, 1, self.config.aqi, self.config.aqf, "const", 1),
            "num_pe": (self.config.num_pe, 0, 16, 0, "const", 1),
            "aqi": (self.config.aqi, 0, 16, 0, "const", 1),
            "aqf": (self.config.aqf, 0, 16, 0, "const", 1),
            "wqi": (self.config.wqi, 0, 16, 0, "const", 1),
            "wqf": (self.config.wqf, 0, 16, 0, "const", 1),
        }
        utils.gen_xilinx_sdk_clib(self.sdk_dir, "edgedrnn_params", sdk_parameters)

    def collect_test_data(self) -> None:
        if self.edgedrnn_params is None:
            raise RuntimeError("collect_params() must run before collect_test_data()")
        print("Collecting simulator stimuli and golden outputs")

        stimulus_batches: list[torch.Tensor] = []
        for batch_index, (features, _) in enumerate(self.dataloaders.test_loader):
            if batch_index == 0:
                stimulus_batches.append(features[0, :, :])
                stimulus_batches.append(features[1:, -1, :])
            else:
                stimulus_batches.append(features[:, -1, :])
        stimulus = torch.cat(stimulus_batches, dim=0)
        stimulus = stimulus[
            self.config.stim_head : self.config.stim_head + self.config.stim_len, :
        ]
        if stimulus.shape[0] != self.config.stim_len:
            raise ValueError(
                f"Requested {self.config.stim_len} stimuli at offset "
                f"{self.config.stim_head}, but only {stimulus.shape[0]} are available"
            )

        with torch.inference_mode():
            model_input = stimulus.unsqueeze(0)
            golden_fc, golden_rnn = self.model(model_input)
            stimulus_array = model_input.squeeze(0).cpu().numpy().transpose()
            golden_fc_array = golden_fc.squeeze(0).cpu().numpy().transpose()
            golden_rnn_array = golden_rnn.squeeze(0).cpu().numpy().transpose()
            if golden_fc_array.ndim == 1:
                golden_fc_array = np.expand_dims(golden_fc_array, axis=0)

            debug_log: dict[str, np.ndarray] = {}
            for layer_index, layer_log in enumerate(self.model.list_log):
                for key, value in layer_log.items():
                    debug_log[f"l{layer_index}_{key}"] = np.transpose(
                        np.squeeze(value.detach().cpu().numpy())
                    )

        input_padding = np.zeros(
            (self.config.inp_size_hw - self.config.inp_size, self.config.stim_len)
        )
        stimulus_array = np.concatenate((stimulus_array, input_padding), axis=0)

        sdk_test_data = {
            "edgedrnn_test_stim": (
                stimulus_array,
                1,
                self.config.aqi,
                self.config.aqf,
                "tensor",
                stimulus_array.shape[0],
            ),
            "edgedrnn_test_gold_fc": (
                golden_fc_array,
                0,
                0,
                0,
                "tensor",
                golden_fc_array.shape[0],
            ),
            "edgedrnn_test_gold_rnn": (
                golden_rnn_array,
                1,
                self.config.aqi,
                self.config.aqf,
                "tensor",
                golden_rnn_array.shape[0],
            ),
        }
        utils.gen_xilinx_sdk_clib(self.sdk_dir, "edgedrnn_test", sdk_test_data)

        config_registers = np.zeros((32, 1))
        config_registers[0] = 0
        config_registers[1] = self.config.thx
        config_registers[2] = self.config.thh
        config_registers[3] = self.config.rnn_layers
        config_registers[4] = self.config.inp_size_hw
        for layer_index in range(self.config.rnn_layers):
            config_registers[5 + layer_index] = self.config.rnn_size

        utils.gen_sim_file(
            self.stimuli_dir,
            {"edgedrnn_tb_params": self.edgedrnn_params},
            self.config.wqi,
            self.config.wqf,
            row_blocks=self.config.num_pe,
        )
        utils.gen_sim_file(
            self.stimuli_dir,
            {
                "edgedrnn_tb_stim": stimulus_array,
                "edgedrnn_tb_gold_rnn": golden_rnn_array,
            },
            self.config.aqi,
            self.config.aqf,
            row_blocks=self.config.num_pe,
        )
        utils.gen_sim_file(
            self.stimuli_dir,
            {"edgedrnn_tb_creg": config_registers},
            16,
            0,
        )

        matlab_parameters = {
            "rnn_param": self.edgedrnn_params,
            "rnn_layers": self.config.rnn_layers,
            "inp_size": self.config.inp_size_hw,
            "rnn_size": self.config.rnn_size,
            "thx": self.config.thx,
            "thh": self.config.thh,
            "num_pe": self.config.num_pe,
            "aqi": self.config.aqi,
            "aqf": self.config.aqf,
            "wqi": self.config.wqi,
            "wqf": self.config.wqf,
        }
        self._save_mat(self.sdk_dir / "edgedrnn_params.mat", matlab_parameters)
        self._save_mat(self.sdk_dir / "edgedrnn_log.mat", debug_log)

    @staticmethod
    def _save_mat(path: Path, data: dict[str, np.ndarray | int | float]) -> None:
        """Write a MATLAB 5 file with a reproducible descriptive header."""
        io.savemat(path, data)
        description = (
            b"MATLAB 5.0 MAT-file, Platform: EdgeDRNN, Generated by scipy.io.savemat"
        )
        with path.open("r+b") as matlab_file:
            matlab_file.write(description.ljust(116, b" "))

    def generate_luts(self) -> None:
        print("Generating activation lookup tables")
        sigmoid_min = int(utils.cast_fp_to_fxp(-8, self.config.aqi, self.config.aqf))
        sigmoid_max = int(utils.cast_fp_to_fxp(8, self.config.aqi, self.config.aqf))
        sigmoid_input = (
            np.arange(sigmoid_min, sigmoid_max).astype(float) / 2**self.config.aqf
        )
        sigmoid_output = torch.sigmoid(torch.from_numpy(sigmoid_input)).numpy()
        sigmoid_output = utils.quantize_array(
            sigmoid_output, self.config.afqi, self.config.afqf, enable=1
        )
        with (self.stimuli_dir / "sigmoid_lut.txt").open(
            "w", encoding="utf-8"
        ) as output:
            for input_value, output_value in zip(sigmoid_input, sigmoid_output):
                output.write(
                    f"{int(utils.cast_fp_to_fxp(input_value, self.config.aqi, self.config.aqf))} "
                    f": lut_out[i] = "
                    f"{int(utils.cast_fp_to_fxp(output_value, self.config.aqi, self.config.aqf))};\n"
                )

        tanh_min = int(utils.cast_fp_to_fxp(-4, self.config.aqi, self.config.aqf))
        tanh_max = int(utils.cast_fp_to_fxp(4, self.config.aqi, self.config.aqf))
        tanh_input = np.arange(tanh_min, tanh_max).astype(float) / 2**self.config.aqf
        tanh_output = torch.tanh(torch.from_numpy(tanh_input)).numpy()
        tanh_output = utils.quantize_array(
            tanh_output, self.config.afqi, self.config.afqf, enable=1
        )
        with (self.stimuli_dir / "tanh_lut.txt").open("w", encoding="utf-8") as output:
            for input_value, output_value in zip(tanh_input, tanh_output):
                output.write(
                    f"{int(utils.cast_fp_to_fxp(input_value, self.config.aqi, self.config.aqf))} "
                    f": lut_out[i] = "
                    f"{int(utils.cast_fp_to_fxp(output_value, self.config.aqi, self.config.aqf))};\n"
                )

    def write_stimuli_manifest(self) -> None:
        files = {
            "edgedrnn_tb_creg.txt": "EDGEDRNN_TB_NUM_CREG",
            "edgedrnn_tb_gold_rnn.txt": "EDGEDRNN_TB_NUM_GOLD",
            "edgedrnn_tb_params.txt": "EDGEDRNN_TB_NUM_PARAMS",
            "edgedrnn_tb_stim.txt": "EDGEDRNN_TB_NUM_STIM",
        }
        entry_counts = {
            filename: sum(
                1
                for line in (self.stimuli_dir / filename)
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            )
            for filename in files
        }
        manifest = {
            "format_version": 1,
            "model": {
                "input_size": self.config.inp_size_hw,
                "num_pe": self.config.num_pe,
                "rnn_layers": self.config.rnn_layers,
                "rnn_size": self.config.rnn_size,
                "stimulus_timesteps": self.config.stim_len,
            },
            "files": {
                filename: {
                    "entries": entry_counts[filename],
                    "sha256": hashlib.sha256(
                        (self.stimuli_dir / filename).read_bytes()
                    ).hexdigest(),
                }
                for filename in files
            },
        }
        (self.stimuli_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        header_lines = [
            "`ifndef EDGEDRNN_TB_CONFIG_SVH",
            "`define EDGEDRNN_TB_CONFIG_SVH",
            "",
            "// Generated by python/edgedrnn/exporter.py. Do not edit manually.",
        ]
        for filename, macro in files.items():
            header_lines.append(f"`define {macro} {entry_counts[filename]}")
        header_lines.extend(
            (
                (
                    f"`define EDGEDRNN_TB_RESPONSES_PER_TIMESTEP "
                    f"{self.config.rnn_size // self.config.num_pe}"
                ),
                "",
                "`endif",
                "",
            )
        )
        (self.stimuli_dir / "edgedrnn_tb_config.svh").write_text(
            "\n".join(header_lines), encoding="utf-8"
        )


# Preserve the original public class name.
EdgeDRNN = EdgeDRNNExporter


def export_model(config: ProjectConfig) -> None:
    EdgeDRNNExporter(config).export()
