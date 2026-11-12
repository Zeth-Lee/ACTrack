from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


BPE_PATH = "third_party/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
PROFILES = ("b224", "l384", "got10k_b224", "got10k_l384", "all")


def file_sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def matches_profile(required_by, profile):
    if profile == "all" or not required_by:
        return True
    for label in required_by:
        name = Path(label).stem.lower()
        if name in ("all", profile):
            return True


        if profile in ("b224", "l384") and name.startswith(profile + "_"):
            return True
    return False


def load_assets(manifest):
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("assets"), list) or not data["assets"]:
        raise ValueError("The manifest must contain a nonempty assets list")
    identifiers = set()
    paths = set()
    for entry in data["assets"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not entry["id"]:
            raise ValueError("Each asset must have a nonempty string id")
        if entry["id"] in identifiers:
            raise ValueError("Duplicate asset id: " + entry["id"])
        identifiers.add(entry["id"])
        name = entry.get("path")
        if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Asset paths must be repository-relative")
        if name in paths:
            raise ValueError("Duplicate asset path: " + name)
        paths.add(name)
        expected_size = entry.get("bytes")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
            raise ValueError("Asset byte length must be a nonnegative integer: " + entry["id"])
        expected_hash = entry.get("sha256")
        if expected_hash is not None and (not isinstance(expected_hash, str) or len(expected_hash) != 64
                                         or any(c not in "0123456789abcdef" for c in expected_hash)):
            raise ValueError("Invalid SHA256: " + entry["id"])
        required_by = entry.get("required_by", [])
        if not isinstance(required_by, list) or not all(isinstance(x, str) for x in required_by):
            raise ValueError("required_by must be a list of profile or configuration names")
    return data["assets"]


def inspect_asset(root, entry, verify_sha256):
    result = {"id": entry["id"], "path": entry["path"], "status": "ok"}
    path = root / entry["path"]
    if not path.is_file():
        result["status"] = "missing"
        return result
    result["bytes"] = path.stat().st_size
    result["expected_bytes"] = entry["bytes"]
    if result["bytes"] != entry["bytes"]:
        result["status"] = "size_mismatch"
        return result
    result["verification"] = "existence_and_size"
    if verify_sha256:
        expected = entry.get("sha256")
        if expected is None:
            result["status"] = "hash_unavailable"
            return result
        result["sha256"] = file_sha256(path)
        result["expected_sha256"] = expected
        result["verification"] = "sha256"
        if result["sha256"] != expected:
            result["status"] = "hash_mismatch"
    return result


def check_assets(root, manifest, profile="all", verify_sha256=False):
    root = Path(root).expanduser().resolve()
    entries = load_assets(Path(manifest))
    chosen = [entry for entry in entries if matches_profile(entry.get("required_by", []), profile)]
    if not chosen:
        raise ValueError("No assets match profile: " + profile)
    checks = [inspect_asset(root, entry, verify_sha256) for entry in chosen]
    if BPE_PATH not in {entry["path"] for entry in chosen}:
        checks.append({"id": "sam3_bpe_vocabulary", "path": BPE_PATH,
                       "verification": "existence",
                       "status": "ok" if (root / BPE_PATH).is_file() else "missing"})
    failures = sum(result["status"] != "ok" for result in checks)
    return {"status": "passed" if failures == 0 else "failed", "profile": profile,
            "sha256_requested": verify_sha256, "checked": len(checks),
            "failed": failures, "assets": checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="ACTrack repository root")
    parser.add_argument("--manifest", type=Path, default=Path("configs/assets.json"),
                        help="Asset manifest, relative to --root unless absolute")
    parser.add_argument("--profile", choices=PROFILES, default="all")
    parser.add_argument("--sha256", action="store_true", help="Also hash every selected asset")
    parser.add_argument("--json", action="store_true", help="Print the verification report as JSON")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    try:
        report = check_assets(root, manifest, args.profile, args.sha256)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = {"status": "error", "error": str(error)}
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("Asset check error: " + str(error), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for item in report["assets"]:
            print(item["status"].upper() + " " + item["id"] + ": " + item["path"])
        print(f"{report['checked']} checked, {report['failed']} failed ({args.profile})")
        if not args.sha256:
            print("Sizes checked; use --sha256 to verify file contents.")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
