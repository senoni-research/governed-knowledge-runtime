#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from gkr.m1_authoring import build_review_binding_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build content-addressed bindings for one M1 semantic-review v2 "
            "fragment. Output contains case IDs and hashes only."
        )
    )
    parser.add_argument("oracle_drafts")
    parser.add_argument(
        "--split",
        required=True,
        choices=("development", "validation", "test"),
    )
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="Include one case. Repeat for a partial fragment; omit for the full split.",
    )
    args = parser.parse_args()
    manifest = build_review_binding_manifest(
        args.oracle_drafts,
        split=args.split,
        case_ids=args.case_ids,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
