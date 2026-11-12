from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sequence_names(dataset, selected=None):
    dataset = Path(dataset).resolve()
    listing = dataset / "list.txt"
    if listing.is_file():
        names = [line.strip() for line in listing.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        names = sorted(path.name for path in dataset.iterdir()
                       if path.is_dir() and (path / "groundtruth.txt").is_file())
    if not names or len(set(names)) != len(names):
        raise ValueError("DiDi sequence list must be nonempty and contain no duplicates")
    for name in names:
        if name in (".", "..") or "/" in name or "\\" in name:
            raise ValueError("DiDi list must contain sequence directory names")
        if not (dataset / name / "groundtruth.txt").is_file() or not (dataset / name / "color").is_dir():
            raise FileNotFoundError(f"Incomplete DiDi sequence: {name}")
    if selected:
        if selected not in names:
            raise ValueError(f"Unknown DiDi sequence: {selected}")
        names = [selected]
    return names


def prepare_workspace(root, dataset, workspace, config, tracker_id, names):
    root, dataset, workspace = (Path(path).resolve() for path in (root, dataset, workspace))
    config = Path(config).resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    stack = root / "didi" / "baseline_experiment.yaml"
    if not stack.is_file():
        raise FileNotFoundError(stack)
    workspace.mkdir(parents=True, exist_ok=False)
    (workspace / "results").mkdir()
    (workspace / "sequences").mkdir()
    for name in names:
        (workspace / "sequences" / name).symlink_to(dataset / name, target_is_directory=True)
    (workspace / "sequences" / "list.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    (workspace / "config.yaml").write_text(
        "registry:\n- ./trackers.ini\nstack: ./baseline_experiment.yaml\n", encoding="utf-8")
    shutil.copy2(stack, workspace / "baseline_experiment.yaml")
    (workspace / "trackers.ini").write_text(f"""[{tracker_id}]
label = ACTrack DiDi
protocol = python
command = actrack.interfaces.didi:create_tracker
arg_repo_root = {root}
arg_config_path = {config}
arg_log_dir = {workspace / 'traces'}
convert = rectangle
timeout = 3600
""", encoding="utf-8")
    jobs = []
    for name in names:
        job = workspace / "jobs" / name
        (job / "sequences").mkdir(parents=True)
        for filename in ("config.yaml", "baseline_experiment.yaml", "trackers.ini"):
            shutil.copy2(workspace / filename, job / filename)
        (job / "sequences" / name).symlink_to(dataset / name, target_is_directory=True)
        (job / "sequences" / "list.txt").write_text(name + "\n", encoding="utf-8")
        (job / "results").symlink_to(workspace / "results", target_is_directory=True)
        jobs.append(job)
    return jobs


def validate_sequence_audit(workspace, name, expected_frames):

    trace = Path(workspace) / "traces" / name
    receipt = json.loads((trace / "AUDIT.json").read_text(encoding="utf-8"))
    if (receipt.get("sequence") != name or receipt.get("complete") is not True
            or receipt.get("frames_expected") != expected_frames
            or receipt.get("frames_recorded") != expected_frames):
        raise RuntimeError(f"DiDi audit is incomplete or mismatched: {name}")
    if (receipt.get("valid_for_reproduction") is not True
            or receipt.get("audit", {}).get("api_failures") != 0):
        raise RuntimeError(f"DiDi audit contains API failures or is invalid: {name}")
    log = trace / "frame_logs.jsonl"
    expected_hash = receipt.get("frame_logs_sha256")
    if expected_hash is not None:
        if hashlib.sha256(log.read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError(f"DiDi frame log hash mismatch: {name}")
        records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        if [row.get("frame_id") for row in records] != list(range(expected_frames)):
            raise RuntimeError(f"DiDi frame log order mismatch: {name}")
    elif log.exists():
        raise RuntimeError(f"DiDi frame log has no completion hash: {name}")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True, help="New output workspace")
    parser.add_argument("--config", type=Path, help="Tracking configuration")
    parser.add_argument("--sequence", help="Run one sequence instead of the entire official list")
    parser.add_argument("--prepare-only", action="store_true", help="Create the workspace without inference")
    parser.add_argument("--score", action="store_true", help="Score after all requested sequences succeed")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    data, workspace = args.dataset.resolve(), args.workspace.resolve()
    config = args.config or root / "configs" / "didi" / "l384_full.yaml"
    names = sequence_names(data, args.sequence)
    tracker_id = "actrack-didi-full"
    jobs = prepare_workspace(root, data, workspace, config, tracker_id, names)
    if args.prepare_only:
        print(f"Prepared {len(jobs)} sequences in {workspace}")
        return
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), str(root), environment.get("PYTHONPATH", "")])
    environment.setdefault("OMP_NUM_THREADS", "4")
    environment.setdefault("MKL_NUM_THREADS", "4")
    completed = []
    for name, job in zip(names, jobs):
        subprocess.run([sys.executable, "-m", "vot", "evaluate", "--workspace", str(job), tracker_id],
                       cwd=root, env=environment, check=True)
        expected_frames = sum(1 for path in (data / name / "color").iterdir()
                              if path.suffix in {".jpg", ".jpeg", ".JPG", ".JPEG"})
        completed.append(validate_sequence_audit(workspace, name, expected_frames))
    receipt = {"tracker": tracker_id, "sequences": names, "completed": completed,
               "frames": sum(row["frames_recorded"] for row in completed),
               "valid_for_reproduction": True,
               "config_sha256": hashlib.sha256(Path(config).read_bytes()).hexdigest()}
    if args.score:
        subprocess.run([sys.executable, "-m", "vot", "analysis", "--workspace", str(workspace),
                        "--format=json", "--name=didi_results", tracker_id],
                       cwd=root, env=environment, check=True)
    (workspace / "COMPLETE.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
