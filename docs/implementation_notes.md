# Implementation Notes

## CUDA Ideas Converted to OpenMP

The referenced CUDA implementation is an exclusive Blelloch scan. Its important ideas are algorithmic rather than CUDA-specific:

1. Scan local blocks independently.
2. Store one total sum per block.
3. Scan the block sums to obtain block offsets.
4. Add each block offset back to its scanned block output.
5. Pad incomplete work with the identity value `0` so arbitrary input lengths are valid.

In this OpenMP project, a "block" becomes a contiguous CPU chunk owned by an OpenMP thread. The chunked implementation follows the same hierarchy without device memory, kernels, shared memory, or CUDA synchronization primitives.

## Direct Blelloch Version

`openmp_blelloch_scan` is the most direct teaching version.

- Copy input into a padded work vector.
- Up-sweep over tree levels using `#pragma omp for`.
- Set the root to zero for exclusive scan.
- Down-sweep over tree levels using `#pragma omp for`.
- Resize the padded work vector back to the original size.

Each tree level has an implicit barrier at the end of the OpenMP loop. That barrier is required because the next level depends on the previous level's results.

Inclusive scan is produced by adding the original input elementwise to the exclusive result.

## Chunked n > p Version

`openmp_chunked_scan` is the practical CPU version.

1. Split the input into approximately equal contiguous chunks.
2. Each thread computes a sequential scan of its own chunk.
3. Each thread writes its chunk total into a padded sum slot to reduce false sharing.
4. The chunk totals are scanned with the direct Blelloch implementation.
5. Each thread adds its chunk offset to every local output element.

This version reduces the number of full-array global synchronization points and usually behaves better for large `n`.

## Correctness Strategy

The sequential scan is the reference for both exclusive and inclusive modes. The test suite checks:

- empty input
- single element input
- small arrays with zeros
- power-of-two input sizes
- non-power-of-two input sizes
- larger arrays
- multiple thread counts

The benchmark path also verifies every parallel result before reporting speedup and efficiency.

## Expected Bottlenecks

- Direct Blelloch: synchronization between every tree level.
- Chunked scan: memory bandwidth and the final offset-add pass.
- Small arrays: OpenMP thread overhead can dominate runtime.
- Large arrays: cache behavior and memory traffic dominate more than arithmetic.

## Notes for Viva

The direct version is best for explaining Blelloch's up-sweep/down-sweep algorithm. The chunked version is best for explaining how GPU-style block sums become a shared-memory CPU hierarchy.

The implementation is deliberately not overloaded with advanced tricks. It uses static scheduling, contiguous vectors, zero padding, and a small false-sharing reduction for chunk sums.
