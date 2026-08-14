#!/usr/bin/env python3
"""Validate generated RTL stimuli before simulator compilation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

MACROS = {
    "edgedrnn_tb_creg.txt": "EDGEDRNN_TB_NUM_CREG",
    "edgedrnn_tb_gold_rnn.txt": "EDGEDRNN_TB_NUM_GOLD",
    "edgedrnn_tb_params.txt": "EDGEDRNN_TB_NUM_PARAMS",
    "edgedrnn_tb_stim.txt": "EDGEDRNN_TB_NUM_STIM",
}


def validate(stimuli_dir: Path) -> None:
    manifest_path = stimuli_dir / "manifest.json"
    config_path = stimuli_dir / "edgedrnn_tb_config.svh"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = config_path.read_text(encoding="utf-8")

    if manifest.get("format_version") != 1:
        raise ValueError(f"Unsupported stimulus manifest: {manifest_path}")

    for filename, macro in MACROS.items():
        path = stimuli_dir / filename
        metadata = manifest["files"][filename]
        entry_count = sum(
            1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if entry_count != metadata["entries"]:
            raise ValueError(
                f"{filename}: expected {metadata['entries']} entries, found {entry_count}"
            )
        if digest != metadata["sha256"]:
            raise ValueError(f"{filename}: SHA-256 does not match manifest.json")

        match = re.search(rf"^`define\s+{macro}\s+(\d+)\s*$", config, re.MULTILINE)
        if match is None or int(match.group(1)) != entry_count:
            raise ValueError(f"{filename}: {macro} does not match the generated file")

    print(f"Validated RTL stimuli in {stimuli_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stimuli_dir", type=Path)
    args = parser.parse_args()
    validate(args.stimuli_dir.resolve())


if __name__ == "__main__":
    main()
