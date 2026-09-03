#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from gkr.m1_authoring import build_dedup_candidates


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic lexical cross-split duplicate-candidate report "
            "outside the repository. Exact normalized cross-split duplicates fail. "
            "The report does not claim semantic review."
        )
    )
    parser.add_argument("cases")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_dedup_candidates(args.cases, output_path=args.output)
    print(
        json.dumps(
            {
                "output": args.output,
                "case_count": report["case_count"],
                "scenario_count": report["scenario_count"],
                "question_set_sha256": report["question_set_sha256"],
                "candidate_count": len(report["candidates"]),
                "exact_cross_split_duplicates": report["exact_cross_split_duplicates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
