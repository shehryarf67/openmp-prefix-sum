# OpenMP Blelloch Prefix Sum

This project is a C++17/OpenMP implementation of parallel prefix sum, also
known as scan. It was prepared for a Parallel and Distributed Computing project
using the CUDA repository `mark-poscablo/gpu-prefix-sum` as a conceptual
reference only.

The final code is pure C++ with OpenMP. No CUDA kernels or CUDA-specific code
are used.

## Blelloch Scan

Prefix sum transforms an input array into running totals. For addition, the two
common forms are:

```text
input:      [3, 1, 7, 0, 4]
exclusive:  [0, 3, 4, 11, 11]
inclusive:  [3, 4, 11, 11, 15]
```

Blelloch scan is a work-efficient parallel exclusive scan with two tree phases:

1. Up-sweep / reduce: build partial sums up to the root.
2. Down-sweep: set the root to the identity value `0`, then distribute prefix
   offsets back down the tree.

Non-power-of-two arrays are handled by padding internally to the next power of
two with zeros. The returned output is resized back to the original input
length.

## Implementations

- `sequential_exclusive_scan` and `sequential_inclusive_scan`: CPU reference
  versions used for correctness.
- `openmp_blelloch_scan`: direct textbook-style OpenMP Blelloch scan over a
  padded tree.
- `openmp_chunked_scan`: practical `n > p` OpenMP version where each thread
  scans a local chunk, one thread scans the small chunk-total array, and all
  threads add their offsets.

The chunked version mirrors the important CUDA baseline ideas in OpenMP form:
block-level/local scan, block sums, hierarchical scan of sums, and uniform add
of block offsets. In this CPU/OpenMP project it is the main practical
implementation because it does most work locally per thread and uses only two
necessary synchronization points. The direct Blelloch version remains useful as
the clearer theoretical tree implementation, but it synchronizes at every tree
level and is usually slower on shared-memory CPUs.

## Project Structure

```text
src/
  main.cpp        CLI, correctness tests, benchmark harness
  scan.cpp        sequential and OpenMP scan implementations
  scan.h          public scan API
  scan.hpp        compatibility include wrapper

scripts/
  run_benchmarks.sh
  plot_results.py

charts/
  speedup_*.svg
  efficiency_*.svg

docs/
  implementation_notes.md

Makefile
README.md
results.csv
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

Sample output from this machine:

```text
./prefix_scan --test
Correctness tests: PASS (160 checks)
```

The tests compare both OpenMP implementations against the sequential reference
for exclusive and inclusive scans. Covered cases include empty input, single
element input, small arrays, power-of-two sizes, non-power-of-two sizes, a
larger array, and multiple thread counts.

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

One benchmark example:

```text
$ ./prefix_scan --n 1000000 --threads 4 --repeats 5 --mode chunked --scan-type exclusive
Mode: chunked
Scan type: exclusive
Input size: 1000000
Threads: 4
Repeats: 5
Sequential average time (ms): 1.9656
Parallel average time (ms):   2.7694
Speedup:                      0.7097
Efficiency:                   0.1774
Correctness:                  PASS
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

Run the full benchmark sweep and generate charts:

```bash
make benchmark
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
python3 scripts/plot_results.py results.csv
```

For a larger memory-bandwidth stress test, use an override such as
`SIZES="10000 100000 1000000 10000000 50000000"` if the machine has enough
available memory.

The plotting script uses only the Python standard library and writes SVG files
to `charts/`:

```text
charts/speedup_chunked_exclusive.svg
charts/efficiency_chunked_exclusive.svg
charts/speedup_chunked_inclusive.svg
charts/efficiency_chunked_inclusive.svg
charts/speedup_direct_exclusive.svg
charts/efficiency_direct_exclusive.svg
charts/speedup_direct_inclusive.svg
charts/efficiency_direct_inclusive.svg
```

## Benchmark Results

The checked-in `results.csv` was generated on 2026-05-09 on:

```text
Intel(R) Core(TM) i5-7300U CPU @ 2.60GHz
4 logical CPUs, Linux 6.17.0-23-generic x86_64
```

The full sweep produced 64 successful benchmark rows. The best observed row was
the chunked exclusive scan at `n=10000000` and 1 OpenMP thread:

```text
sequential_ms=28.4287, parallel_ms=26.7592, speedup=1.0624, efficiency=1.0624
```

For larger inputs, the chunked implementation was consistently much faster than
the direct tree implementation, but scaling across more threads was limited on
this 2-core/4-thread CPU. Representative rows from `results.csv`:

| Mode | Scan | n | Threads | Sequential ms | Parallel ms | Speedup | Efficiency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| direct | exclusive | 10000000 | 4 | 25.2433 | 225.3693 | 0.1120 | 0.0280 |
| direct | inclusive | 10000000 | 4 | 24.7321 | 215.6207 | 0.1147 | 0.0287 |
| chunked | exclusive | 10000000 | 4 | 23.9236 | 29.0129 | 0.8246 | 0.2061 |
| chunked | inclusive | 10000000 | 4 | 25.5366 | 37.7018 | 0.6773 | 0.1693 |

These results show that the direct Blelloch version is useful for explaining the
algorithm, while the chunked version is the better practical OpenMP design.

## Limitations

- Large scans are mostly memory-bound: the code streams through large vectors
  and performs very little arithmetic per element.
- The direct Blelloch implementation synchronizes at every tree level, so
  barrier overhead is high relative to the work per level.
- The chunked version reduces synchronization by doing local work inside one
  OpenMP parallel region, but it still needs a final offset-add pass.
- The CUDA project is used only as a conceptual reference. This repository does
  not use CUDA, GPU kernels, shared memory, or device-side synchronization, so
  the timings should not be compared as GPU performance results.

## Academic Integrity Note

The CUDA baseline was used to understand the algorithmic structure: local scan,
block sums, recursive/hierarchical scan of sums, offset addition, and padding
for arbitrary input sizes. This repository reimplements those ideas
independently for a shared-memory CPU environment using OpenMP.

The code is intentionally kept readable for viva discussion and course
evaluation.
