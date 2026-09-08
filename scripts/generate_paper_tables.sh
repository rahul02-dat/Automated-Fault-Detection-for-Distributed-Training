#!/usr/bin/env bash
set -e

echo "Generating paper tables from raw JSON outputs..."

mkdir -p results/summary
uv run python -m src.experiments.analyze_results > results/summary/paper_tables.md

echo "Tables generated at results/summary/paper_tables.md"
cat results/summary/paper_tables.md
