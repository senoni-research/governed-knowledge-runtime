#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m pytest \
  tests/test_schemas.py \
  tests/test_authority_store.py \
  tests/test_proposals.py \
  tests/test_source_integrity.py \
  tests/test_decision.py \
  tests/test_trace.py \
  tests/test_runtime.py \
  tests/test_cli.py \
  tests/test_evaluation.py::test_frozen_m0_suite_passes_against_demo_authority
