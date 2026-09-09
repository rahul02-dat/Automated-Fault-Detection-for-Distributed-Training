#!/usr/bin/env bash
set -e

echo "Generating paper tables from raw JSON outputs..."

mkdir -p results/analysis

uv run python -m src.experiments.analyze_results

echo ""
echo "Tables generated at results/analysis/paper_tables.md"
echo ""

# Also generate aggregate report if results exist
if [ -d "results/raw" ]; then
    echo "Generating aggregate experiment report..."
    uv run python -m src.experiments.experiment_runner --raw-dir results/raw --output results/analysis/aggregate_report.txt
    echo "Aggregate report at results/analysis/aggregate_report.txt"
fi
