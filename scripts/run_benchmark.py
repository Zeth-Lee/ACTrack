from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


def input_path(value, base):
    path = Path(value).expanduser()
    return Path(os.path.abspath(path if path.is_absolute() else base / path))


def load_sequences(manifest, selected=None):
    manifest = Path(manifest).resolve()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    rows = data.get("sequences") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must contain a nonempty sequences list")
    seen = set()
    sequences = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each sequence must be an object")
        name = row.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[\w.-]+", name) or name in (".", ".."):
            raise ValueError("Sequence name must be a single safe directory name")
        if name.casefold() in {"run.json", "complete.json"}:
            raise ValueError("Sequence name conflicts with a benchmark receipt")
        if name in seen:
            raise ValueError(f"Duplicate sequence name: {name}")
        seen.add(name)
        frames = input_path(row["frames"], manifest.parent)
        if not frames.is_dir():
            raise NotADirectoryError(frames)
        has_file, has_box = "init_file" in row, "init_bbox" in row
        if has_file == has_box:
            raise ValueError(f"{name}: supply exactly one init_file or init_bbox")
        entry = {"name": name, "frames": frames}
        if has_file:
            path = input_path(row["init_file"], manifest.parent)
            if not path.is_file():
                raise FileNotFoundError(path)
            entry["init_file"] = path
        else:
            box = row["init_bbox"]
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise ValueError(f"{name}: init_bbox must contain four xywh values")
            values = tuple(float(value) for value in box)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name}: init_bbox contains non-finite coordinates")
            entry["init_bbox"] = values
        sequences.append(entry)
    if selected:
        if selected not in seen:
            raise ValueError(f"Unknown sequence: {selected}")
        sequences = [entry for entry in sequences if entry["name"] == selected]
    return sequences


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSON sequence list; data paths relative to this file or absolute")
    parser.add_argument("--config", type=Path, default=root / "configs/rgb/l384.yaml",
                        help="Tracking configuration (default: configs/rgb/l384.yaml)")
    parser.add_argument("--output", type=Path, required=True, help="New benchmark output directory")
    parser.add_argument("--sequence", help="Run one named sequence from the manifest")
    args = parser.parse_args(argv)
    manifest, config, output = args.manifest.resolve(), args.config.resolve(), args.output.resolve()
    sequences = load_sequences(manifest, args.sequence)
    if not config.is_file():
        raise FileNotFoundError(config)
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "sequences": [entry["name"] for entry in sequences],
        "execution": "serial; one fresh process per sequence; stop on failure; no automatic retry",
    }
    (output / "RUN.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    completed = []
    for entry in sequences:
        destination = output / entry["name"]
        command = [sys.executable, str(root / "scripts" / "run_rgb.py"),
                   "--config", str(config), "--frames", str(entry["frames"]),
                   "--output", str(destination), "--sequence-name", entry["name"]]
        if "init_file" in entry:
            command.extend(["--init-file", str(entry["init_file"])])
        else:
            command.extend(["--init-bbox", *(format(value, ".17g") for value in entry["init_bbox"])])
        subprocess.run(command, check=True)
        receipt = json.loads((destination / "COMPLETE.json").read_text(encoding="utf-8"))
        completed.append({"name": entry["name"], "frames": receipt["frames"],
                          "valid_for_reproduction": receipt.get("valid_for_reproduction", False)})
    summary = {**metadata, "completed": completed, "frames": sum(row["frames"] for row in completed),
               "valid_for_reproduction": all(row["valid_for_reproduction"] for row in completed)}
    (output / "COMPLETE.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
