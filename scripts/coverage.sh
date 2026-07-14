#!/usr/bin/env bash
#
# Single source of the coverage invocation. CI and local devs both run this
# script, so the lcov Codecov ingests and the numbers you see locally derive
# from one identical run — no invocation drift.
#
# Consumers:
#   - CI uploads target/llvm-cov/lcov.info to Codecov for the PR delta.
#   - Locally, read the printed summary, or `scripts/coverage.sh --html` for a
#     drill-down report under target/llvm-cov/html/.
#
# Runs on *stable* with region+line coverage (no --branch): --branch needs
# nightly and, combined with async-trait's instantiation shape, trips
# LLVM #119558 (a SIGSEGV in `llvm-cov export`). Region+line is the metric
# Codecov headlines regardless.
#
# --all-features pulls in the llama-gguf e2e test, which downloads a ~150 MB
# GGUF model on first run (cached thereafter).
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p target/llvm-cov

# Collect once (instrumented build + full test run); derive every report
# format from the same profile data without rebuilding.
cargo llvm-cov --all-features --no-report test

cargo llvm-cov report --lcov --output-path target/llvm-cov/lcov.info

if [[ "${1:-}" == "--html" ]]; then
  cargo llvm-cov report --html
fi

# Printed last so the headline %s are the final thing on screen locally.
cargo llvm-cov report --summary-only
