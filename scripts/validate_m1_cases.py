#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from gkr.m1_validation import validate_m1_cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate governed M1 benchmark JSONL cases")
    parser.add_argument("cases")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Validate cases while the three 120-scenario splits are still being assembled",
    )
    args = parser.parse_args()
    report = validate_m1_cases(args.cases, allow_incomplete=args.allow_incomplete)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
