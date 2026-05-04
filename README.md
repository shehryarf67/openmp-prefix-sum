# OpenMP Blelloch Prefix Sum

This project is a C++17/OpenMP implementation of parallel prefix sum, also known as scan. It was prepared for a Parallel and Distributed Computing project using the CUDA repository `mark-poscablo/gpu-prefix-sum` as a conceptual reference only.

The final code is pure C++ with OpenMP. No CUDA kernels or CUDA-specific code are used.

## Blelloch Scan

Prefix sum transforms an input array into running totals. For addition, the two common forms are:

```text
input:      [3, 1, 7, 0, 4]
exclusive:  [0, 3, 4, 11, 11]
inclusive:  [3, 4, 11, 11, 15]
```

Blelloch scan is a work-efficient parallel exclusive scan with two tree phases:

1. Up-sweep / reduce: build partial sums up to the root.
2. Down-sweep: set the root to the identity value `0`, then distribute prefix offsets back down the tree.

Non-power-of-two arrays are handled by padding internally to the next power of two with zeros. The returned output is resized back to the original input length.

## Implementations

- `sequential_exclusive_scan` and `sequential_inclusive_scan`: CPU reference versions used for correctness.
- `openmp_blelloch_scan`: direct textbook-style OpenMP Blelloch scan over a padded tree.
- `openmp_chunked_scan`: practical `n > p` OpenMP version where each thread scans a local chunk, chunk sums are scanned, and offsets are added back.

The chunked version mirrors the important CUDA baseline ideas in OpenMP form: block-level/local scan, block sums, hierarchical scan of sums, and uniform add of block offsets.

## Project Structure

```text
src/
  main.cpp        CLI, correctness tests, benchmark harness
  scan.cpp        sequential and OpenMP scan implementations
  scan.h          public scan API
  scan.hpp        compatibility include wrapper

scripts/
  run_benchmarks.sh

docs/
  implementation_notes.md

Makefile
README.md
```

## Build

Requires Linux or WSL with a C++17 compiler and OpenMP support.

```bash
make
```

The Makefile uses:

```bash
g++ -O3 -std=c++17 -fopenmp
```

## Correctness Tests

Run the built-in edge-case test suite:

```bash
make test
```

or:

```bash
./prefix_scan --test
```

The tests compare both OpenMP implementations against the sequential reference for exclusive and inclusive scans. Covered cases include empty input, single element input, small arrays, power-of-two sizes, non-power-of-two sizes, a larger array, and multiple thread counts.

## Run Examples

Run both OpenMP versions with exclusive scan:

```bash
./prefix_scan --n 1000000 --threads 4 --repeats 5 --mode both --scan-type exclusive
```

Run the chunked inclusive version:

```bash
./prefix_scan --n 1000000 --threads 8 --repeats 5 --mode chunked --scan-type inclusive
```

Run only the sequential baseline:

```bash
./prefix_scan --n 1000000 --mode sequential --scan-type exclusive
```

## Command-Line Options

```text
--n <input size>
--threads <OpenMP threads>
--repeats <timed repetitions>
--mode <sequential|direct|chunked|both>
--scan-type <exclusive|inclusive>
--csv
--test
```

Timed runs report average time across repetitions.

## Metrics

For parallel modes, the program reports:

- `sequential_ms`: average sequential reference time.
- `parallel_ms`: average OpenMP time.
- `speedup`: `sequential_ms / parallel_ms`.
- `efficiency`: `speedup / thread_count`.
- `status`: correctness result against the sequential baseline.

## Benchmark Script

Run the full benchmark sweep:

```bash
./scripts/run_benchmarks.sh
```

By default, it runs:

- input sizes: `10000 100000 1000000 10000000`
- thread counts: `1 2 4 8`
- modes: `direct chunked`
- scan types: `exclusive inclusive`

Results are saved to `results.csv` with:

```text
n,threads,scan_type,mode,sequential_ms,parallel_ms,speedup,efficiency,status
```

You can override the sweep without editing the script:

```bash
SIZES="10000 100000" THREADS="1 2 4" REPEATS=5 ./scripts/run_benchmarks.sh
```

## Academic Integrity Note

The CUDA baseline was used to understand the algorithmic structure: local scan, block sums, recursive/hierarchical scan of sums, offset addition, and padding for arbitrary input sizes. This repository reimplements those ideas independently for a shared-memory CPU environment using OpenMP.

The code is intentionally kept readable for viva discussion and course evaluation.
