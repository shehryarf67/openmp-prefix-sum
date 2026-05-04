#!/usr/bin/env bash
set -euo pipefail

make

RESULTS_FILE="${RESULTS_FILE:-results.csv}"
SIZES="${SIZES:-10000 100000 1000000 10000000}"
THREADS="${THREADS:-1 2 4 8}"
REPEATS="${REPEATS:-3}"
SCAN_TYPES="${SCAN_TYPES:-exclusive inclusive}"
MODES="${MODES:-direct chunked}"

echo "n,threads,scan_type,mode,sequential_ms,parallel_ms,speedup,efficiency,status" > "$RESULTS_FILE"

for scan_type in $SCAN_TYPES; do
  for mode in $MODES; do
    for n in $SIZES; do
      for t in $THREADS; do
        ./prefix_scan \
          --n "$n" \
          --threads "$t" \
          --repeats "$REPEATS" \
          --mode "$mode" \
          --scan-type "$scan_type" \
          --csv | tail -n +2 >> "$RESULTS_FILE"
      done
    done
  done
done

echo "Wrote $RESULTS_FILE"
