#!/usr/bin/env python3
"""Deterministic candidate and selector fixture for HLoop parity tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    candidate = subparsers.add_parser("candidate")
    candidate.add_argument("--candidate-id", required=True)
    candidate.add_argument("--seed", type=int, required=True)
    candidate.add_argument("--score", type=float, required=True)
    candidate.add_argument("--input", type=Path, required=True)
    candidate.add_argument("--output", type=Path, required=True)
    candidate.add_argument("--sleep", type=float, default=0.0)
    selector = subparsers.add_parser("select")
    selector.add_argument("--candidate", type=Path, action="append", required=True)
    selector.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "candidate":
        time.sleep(args.sleep)
        source_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
        payload = {
            "candidate_id": args.candidate_id,
            "seed": args.seed,
            "score": args.score,
            "source_sha256": source_hash,
            "cuda_visible_devices": os.environ.get("MIMICX_DEVICE_ID"),
        }
    else:
        candidates = [json.loads(path.read_text(encoding="utf-8")) for path in args.candidate]
        selected = max(candidates, key=lambda item: (float(item["score"]), str(item["candidate_id"])))
        payload = {
            "candidate_ids": sorted(str(item["candidate_id"]) for item in candidates),
            "selected_id": selected["candidate_id"],
            "selected_score": selected["score"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
