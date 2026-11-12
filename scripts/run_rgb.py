from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys


def parse_bbox(line):
    parts = [part for part in re.split(r"[,\s]+", line.strip()) if part]
    if len(parts) != 4:
        raise ValueError("Expected exactly four xywh values; polygon/mask annotations need their benchmark adapter")
    box = tuple(float(value) for value in parts)
    if not all(math.isfinite(value) for value in box):
        raise ValueError("Initialization box contains non-finite values")
    return box


def first_bbox(path):

    with Path(path).open(encoding="utf-8-sig") as stream:
        line = stream.readline()
    if not line.strip():
        raise ValueError("Initialization file must start with a bounding box")
    return parse_bbox(line)


def frame_sort_key(path):

    key = []
    for part in re.split(r"(\d+)", path.stem.lower()):
        if part:
            key.append((0, int(part)) if part.isdigit() else (1, part))
    key.append((2, path.suffix.lower()))
    return tuple(key)


def ordered_frames(directory=None, listing=None):
    if (directory is None) == (listing is None):
        raise ValueError("Specify one frame directory or ordered frame list")
    if directory is not None:
        directory = Path(os.path.abspath(Path(directory).expanduser()))
        frames = sorted((path for path in directory.iterdir()
                         if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")),
                        key=frame_sort_key)
    else:
        listing = Path(os.path.abspath(Path(listing).expanduser()))
        entries = [line.strip() for line in listing.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        frames = [Path(os.path.abspath(Path(entry).expanduser() if Path(entry).expanduser().is_absolute()
                                      else listing.parent / Path(entry).expanduser()))
                  for entry in entries]
    if not frames:
        raise ValueError("No frames found")
    if len(set(frames)) != len(frames):
        raise ValueError("Duplicate frame paths are not allowed")
    for frame in frames:
        if not frame.is_file():
            raise FileNotFoundError(frame)
    return frames


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/rgb/l384.yaml",
                        help="Tracking configuration (default: configs/rgb/l384.yaml)")
    frame_source = parser.add_mutually_exclusive_group(required=True)
    frame_source.add_argument("--frames", type=Path, help="Complete JPEG frame directory with integer filenames")
    frame_source.add_argument("--frame-list", type=Path, help="Complete same-directory JPEG list in numeric order; paths relative to this file")
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--init-bbox", nargs=4, type=float, metavar=("X", "Y", "W", "H"))
    initialization.add_argument("--init-file", type=Path, help="Read only the first xywh annotation line")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing results are not reused")
    parser.add_argument("--sequence-name", help="Optional sequence identifier")
    args = parser.parse_args(argv)
    frames = ordered_frames(args.frames, args.frame_list)
    box = first_bbox(args.init_file) if args.init_file else tuple(args.init_bbox)
    name = args.sequence_name
    if not name:
        name = frames[0].parent.parent.name if frames[0].parent.name in ("img", "imgs", "color") else frames[0].parent.name
    sys.path.insert(0, str(root / "src"))
    from actrack.interfaces.rgb import run_sequence

    receipt = run_sequence(args.config.resolve(), frames, box, args.output,
                           sequence_name=name, repo_root=root)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
