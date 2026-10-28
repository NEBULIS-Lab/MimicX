#!/usr/bin/env python3
"""Audit authorized local motions without GPU, downloads, or policy screening."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mimicx.datasets.motion_study import prepare, freeze_ground_contact_protocol


def main():
    if '--freeze-protocol-from' in sys.argv:
        import json
        p = argparse.ArgumentParser(description='Issue a new protocol-frozen manifest without altering motions')
        p.add_argument('--freeze-protocol-from', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--agreements', type=Path, required=True)
        p.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[2] / 'mimicx/configs/datasets/motion_study_v1.json')
        args = p.parse_args()
        protocol = json.loads(args.config.read_text())['ground_contact_protocol']
        result = freeze_ground_contact_protocol(args.freeze_protocol_from, args.output, protocol,
                                                json.loads(args.agreements.read_text()))
        print(result['counts'])
        print(args.output / 'manifest.json')
        return 2 if result['counts'].get('blocked', 0) else 0
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "mimicx/configs/datasets/motion_study_v1.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--g1-xml", type=Path, required=True)
    p.add_argument("--regenerate-sources", action="store_true")
    for name in ("amass-csv", "gmr-timing-source", "csv-converter-source", "amass-producer-source"):
        p.add_argument("--" + name, type=Path)
    for name in ("bones-root", "amass-root", "lafan-root", "amass-original", "lafan-csv", "bones-metadata"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    roots = {"BONES": args.bones_root, "AMASS": args.amass_root, "LaFAN1": args.lafan_root,
             "amass_original": args.amass_original, "lafan_csv": args.lafan_csv,
             "bones_metadata": args.bones_metadata}
    if args.regenerate_sources:
        for key in ('amass_csv', 'gmr_timing_source', 'csv_converter_source', 'amass_producer_source'):
            if getattr(args, key) is None:
                p.error('--regenerate-sources requires --' + key.replace('_', '-'))
            roots[key] = getattr(args, key)
    result = prepare(args.config, args.output, roots, args.g1_xml, args.regenerate_sources)
    print(result["counts"])
    print(args.output / "manifest.json")
    return 2 if result["counts"].get("blocked", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
