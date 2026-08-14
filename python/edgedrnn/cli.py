"""EdgeDRNN command-line workflows."""

from __future__ import annotations

from collections.abc import Sequence

from edgedrnn.config import ProjectConfig
from edgedrnn.exporter import export_model
from edgedrnn.trainer import train


def _print_stage(title: str) -> None:
    border = "#" * 80
    print(f"{border}\n# {title}\n{border}")


def main(argv: Sequence[str] | None = None) -> None:
    config = ProjectConfig(argv)
    for step in config.selected_steps():
        config.set_step(step)
        if step == "pretrain":
            _print_stage("Pretrain GRU")
            train(config)
        elif step == "retrain":
            _print_stage("Retrain DeltaGRU")
            train(config)
        elif step == "export":
            _print_stage("Export EdgeDRNN hardware assets")
            export_model(config)

        # A user-supplied checkpoint is an input to the selected stage. Later
        # run-through stages must consume the checkpoint just produced.
        if config.run_through and step != "export":
            config.update_attr("model_path", "")


if __name__ == "__main__":
    main()
