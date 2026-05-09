# perf Profiling Notes

Profiling was attempted with Linux `perf` on:

```text
Linux shehryar-Surface-Laptop 6.17.0-23-generic x86_64
```

The kernel currently has:

```text
/proc/sys/kernel/perf_event_paranoid = 4
```

Because of that setting, hardware performance counters such as cycles,
instructions, cache references, cache misses, and branch misses were not
available to this user session. `perf stat -d` failed with the kernel security
message explaining that performance monitoring is restricted.

Tool-level timing counters were still available through:

```bash
env OMP_PROC_BIND=true OMP_PLACES=cores \
perf stat -e duration_time,user_time,system_time \
./prefix_scan --n 10000000 --threads 4 --repeats 9 \
  --mode chunked --scan-type exclusive
```

## Chunked Exclusive, n=10,000,000, 4 Threads

```text
Sequential median time (ms):  8.8337
Parallel median time (ms):    8.8716
Sequential average time (ms): 8.6413
Parallel average time (ms):   11.6882
Speedup:                      0.9957
Efficiency:                   0.2489

duration_time: 609,331,327 ns
user_time:     572,348,000 ns
system_time:   130,483,000 ns
time elapsed:  0.609279785 s
```

## Direct Exclusive, n=10,000,000, 4 Threads

```text
Sequential median time (ms):  7.9894
Parallel median time (ms):    113.3692
Sequential average time (ms): 8.3416
Parallel average time (ms):   118.9867
Speedup:                      0.0705
Efficiency:                   0.0176

duration_time: 1,934,136,259 ns
user_time:     4,054,850,000 ns
system_time:   168,303,000 ns
time elapsed:  1.934086236 s
```

## Interpretation

The chunked implementation is the practical OpenMP path. It completes the same
10M-element benchmark much faster than the direct Blelloch version because it
does most work locally inside each contiguous chunk and only synchronizes around
the thread-count-sized chunk sum/offset phase.

The direct Blelloch version remains useful as a theoretical reference, but its
tree-level barriers make it expensive on a shared-memory CPU. The much higher
user time for the direct run is consistent with threads spending more total CPU
time inside the synchronized tree phases.

For deeper profiling, lower the kernel setting temporarily:

```bash
sudo sysctl kernel.perf_event_paranoid=1
```

Then rerun:

```bash
env OMP_PROC_BIND=true OMP_PLACES=cores \
perf stat -r 3 -d \
./prefix_scan --n 10000000 --threads 4 --repeats 9 \
  --mode chunked --scan-type exclusive
```

That should expose cycles, instructions, IPC, branches, and cache statistics if
the machine grants the required perf permissions.

## Call Graph

`perf record --call-graph dwarf` was also attempted, but it was blocked by the
same `perf_event_paranoid = 4` setting. As a fallback, a separate `gprof`
instrumented binary was built with `-pg`, run on the chunked benchmark, and
converted into a Graphviz call graph:

```text
docs/gprof_chunked.txt
docs/chunked_call_graph.dot
docs/chunked_call_graph.svg
```

This is a call graph, but not a hardware-sampled `perf` flame graph. It is still
useful for explaining the benchmark flow: `main` prepares input and timing,
`sequential_scan_into` builds the correctness reference, and the benchmark
lambda calls `openmp_chunked_scan_into` for the warm-up and timed repetitions.
