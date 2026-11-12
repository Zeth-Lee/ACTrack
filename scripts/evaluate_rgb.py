from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys


def read_boxes(path, *, allow_nan=False):
    from actrack.types import BBox

    boxes = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            parts = [part for part in re.split(r"[,\s]+", line.strip()) if part]
            if len(parts) != 4:
                raise ValueError(f"{path}:{number}: expected four xywh values")
            values = tuple(float(value) for value in parts)
            if not all(math.isfinite(value) or (allow_nan and math.isnan(value)) for value in values):
                raise ValueError(f"{path}:{number}: non-finite coordinates")
            boxes.append(BBox(*values))
    if not boxes:
        raise ValueError(f"Empty trajectory: {path}")
    return boxes


def normalized_precision_at_threshold(preds, gts, dist_thr=0.2):

    n = min(len(preds), len(gts))
    if n == 0:
        return 0.0
    hit = 0
    for i in range(n):
        pred = preds[i]
        gt = gts[i]
        px = pred.x + 0.5 * (pred.w - 1.0)
        py = pred.y + 0.5 * (pred.h - 1.0)
        gx = gt.x + 0.5 * (gt.w - 1.0)
        gy = gt.y + 0.5 * (gt.h - 1.0)
        gw = max(gt.w, 1e-6)
        gh = max(gt.h, 1e-6)
        distance = math.hypot((px - gx) / gw, (py - gy) / gh)
        if distance <= dist_thr:
            hit += 1
    return float(hit / n)


def evaluate(predictions, groundtruth):
    from actrack.eval.metrics import precision_at_threshold, success_auc


    preds, gts = read_boxes(predictions), read_boxes(groundtruth, allow_nan=True)
    if len(preds) != len(gts):
        raise ValueError(f"Frame count mismatch: {len(preds)} predictions, {len(gts)} ground-truth boxes")
    metrics = {
        "suc": success_auc(preds, gts, bins=21),
        "pnorm": normalized_precision_at_threshold(preds, gts, dist_thr=0.2),
        "p": precision_at_threshold(preds, gts, dist_thr=20.0),
    }
    return {
        "frames": len(preds),
        "frame_policy": "all supplied frames including initialization and NaN ground truth; no visibility filtering",
        "metric_definition": "mean success at 21 IoU thresholds [0,1]; normalized precision at 0.2; precision at 20 pixels",
        "metrics_fraction": metrics,
        "metrics_percent": {name: 100.0 * value for name, value in metrics.items()},
    }


def evaluate_manifest(manifest):
    manifest = Path(manifest).expanduser().resolve()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    rows = data.get("sequences") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must contain a nonempty sequences list")
    seen, records = set(), []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each sequence must be an object")
        name = row.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[\w.-]+", name) or name in (".", ".."):
            raise ValueError("Sequence name must be a single safe directory name")
        if name in seen:
            raise ValueError(f"Duplicate sequence name: {name}")
        seen.add(name)
        paths = {}
        for key in ("predictions", "groundtruth"):
            path = Path(row[key]).expanduser()
            paths[key] = (path if path.is_absolute() else manifest.parent / path).resolve()
        records.append({"name": name, **evaluate(**paths)})
    metrics = {name: sum(row["metrics_fraction"][name] for row in records) / len(records)
               for name in ("suc", "pnorm", "p")}
    return {
        "sequences": len(records),
        "frames": sum(row["frames"] for row in records),
        "aggregation": "unweighted arithmetic mean of sequence metrics (macro); every sequence has equal weight",
        "frame_policy": records[0]["frame_policy"],
        "metric_definition": records[0]["metric_definition"],
        "metrics_fraction": metrics,
        "metrics_percent": {name: 100.0 * value for name, value in metrics.items()},
        "per_sequence": records,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, help="Complete xywh trajectory, including frame zero")
    parser.add_argument("--groundtruth", type=Path, help="Scoring annotations; never supplied to inference")
    parser.add_argument("--manifest", type=Path, help="JSON name/predictions/groundtruth list for unweighted sequence aggregation")
    parser.add_argument("--output", type=Path, help="Optional new JSON file")
    args = parser.parse_args(argv)
    if args.manifest:
        if args.predictions is not None or args.groundtruth is not None:
            parser.error("--manifest cannot be combined with --predictions or --groundtruth")
    elif args.predictions is None or args.groundtruth is None:
        parser.error("Supply --manifest or both --predictions and --groundtruth")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    report = evaluate_manifest(args.manifest) if args.manifest else evaluate(args.predictions, args.groundtruth)
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
