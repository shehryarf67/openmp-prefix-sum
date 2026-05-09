#!/usr/bin/env bash
set -euo pipefail

make

RESULTS_FILE="${RESULTS_FILE:-results.csv}"
SIZES="${SIZES:-10000 100000 1000000 10000000}"
THREADS="${THREADS:-1 2 4 8}"
REPEATS="${REPEATS:-9}"
SCAN_TYPES="${SCAN_TYPES:-exclusive inclusive}"
MODES="${MODES:-direct chunked}"

# Recommended for lower-noise OpenMP benchmarking. Users can override either
# variable before running this script.
export OMP_PROC_BIND="${OMP_PROC_BIND:-true}"
export OMP_PLACES="${OMP_PLACES:-cores}"

if [[ "${INCLUDE_50M:-0}" == "1" ]]; then
  SIZES="$SIZES 50000000"
fi

printf "%s%s%s\n" \
  "n,threads,scan_type,mode,sequential_ms,parallel_ms,speedup," \
  "efficiency,status,repeats,timing_method,median_sequential_ms," \
  "median_parallel_ms,average_sequential_ms,average_parallel_ms" \
  > "$RESULTS_FILE"

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
