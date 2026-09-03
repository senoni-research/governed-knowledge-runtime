#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from gkr.m1_oracle_validation import load_case_jsonl
from gkr.m1_validation import validate_m1_cases

_SCHEMA_FILES = {
    "v1": "evaluation/m1/benchmark-case.schema.json",
    "v2": "evaluation/m1/benchmark-case-v2.schema.json",
    "v3": "evaluation/m1/benchmark-case-v3.schema.json",
}
_SCHEMA_CONSTANTS = {
    "v1": "gkr-m1-case-v1",
    "v2": "gkr-m1-case-v2",
    "v3": "gkr-m1-case-v3",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate governed M1 benchmark JSONL cases. Schema selection is "
            "explicit: pass --schema-version or allow detection from the first "
            "case. v3 cases are never silently validated against v1 or v2."
        )
    )
    parser.add_argument("cases")
    parser.add_argument(
        "--schema-version",
        choices=("v1", "v2", "v3"),
        help="Explicit contract version. Required when schema_version cannot be detected.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Validate cases while the three 120-scenario splits are still being assembled",
    )
    args = parser.parse_args()
    schema_version = args.schema_version or _detect_schema_version(args.cases)
    report = validate_m1_cases(
        args.cases,
        schema_path=_SCHEMA_FILES[schema_version],
        allow_incomplete=args.allow_incomplete,
    )
    report["schema_version"] = schema_version
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _detect_schema_version(case_path: str) -> str:
    cases = load_case_jsonl(case_path)
    declared = {str(case.get("schema_version", "")) for _line, case in cases}
    if len(declared) != 1:
        raise ValueError(
            f"{case_path}: mixed or missing schema_version values; pass "
            "--schema-version v1|v2|v3 explicitly. v3 cases must not be "
            "validated against v1 or v2."
        )
    version = next(iter(declared))
    for label, constant in _SCHEMA_CONSTANTS.items():
        if version == constant:
            return label
    raise ValueError(
        f"{case_path}: unrecognized schema_version {version!r}; pass "
        "--schema-version v1|v2|v3 explicitly"
    )


if __name__ == "__main__":
    raise SystemExit(main())
