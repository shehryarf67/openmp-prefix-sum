#include "scan.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <omp.h>

namespace scan {

namespace {

struct ChunkRange {
    std::size_t start;
    std::size_t end;
};

void validate_thread_count(int num_threads) {
    if (num_threads < 1) {
        throw std::invalid_argument("num_threads must be at least 1");
    }
}

void add_input_in_place(
    std::vector<value_t>& output,
    const std::vector<value_t>& input,
    int num_threads
) {
    if (output.empty()) {
        return;
    }

    const auto n = static_cast<std::ptrdiff_t>(output.size());
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        output[static_cast<std::size_t>(i)] += input[static_cast<std::size_t>(i)];
    }
}

ChunkRange chunk_range(std::size_t n, int tid, int thread_count) {
    const auto thread = static_cast<std::size_t>(tid);
    const auto threads = static_cast<std::size_t>(thread_count);

    return {
        (n * thread) / threads,
        (n * (thread + 1)) / threads
    };
}

} // namespace

void ChunkedScanWorkspace::resize(int thread_count) {
    validate_thread_count(thread_count);

    const auto size = static_cast<std::size_t>(thread_count);
    if (chunk_sums.size() != size) {
        chunk_sums.resize(size);
    }
    if (chunk_offsets.size() != size) {
        chunk_offsets.resize(size);
    }
}

std::size_t next_power_of_two(std::size_t n) {
    if (n <= 1) return 1;

    if (n > (std::size_t{1} << (std::numeric_limits<std::size_t>::digits - 1))) {
        throw std::length_error("next power of two would overflow size_t");
    }

    --n;
    for (std::size_t shift = 1; shift < sizeof(std::size_t) * 8; shift <<= 1) {
        n |= n >> shift;
    }
    return n + 1;
}

std::vector<value_t> sequential_exclusive_scan(const std::vector<value_t>& input) {
    std::vector<value_t> output;
    sequential_scan_into(input, output, ScanType::Exclusive);
    return output;
}

std::vector<value_t> sequential_inclusive_scan(const std::vector<value_t>& input) {
    std::vector<value_t> output;
    sequential_scan_into(input, output, ScanType::Inclusive);
    return output;
}

void sequential_scan_into(
    const std::vector<value_t>& input,
    std::vector<value_t>& output,
    ScanType scan_type
) {
    output.resize(input.size());
    value_t running_sum = 0;

    if (scan_type == ScanType::Exclusive) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = running_sum;
            running_sum += input[i];
        }
    } else {
        for (std::size_t i = 0; i < input.size(); ++i) {
            running_sum += input[i];
            output[i] = running_sum;
        }
    }
}

std::vector<value_t> sequential_scan(
    const std::vector<value_t>& input,
    ScanType scan_type
) {
    std::vector<value_t> output;
    sequential_scan_into(input, output, scan_type);
    return output;
}

void openmp_blelloch_scan_into(
    const std::vector<value_t>& input,
    std::vector<value_t>& output,
    int num_threads,
    ScanType scan_type
) {
    validate_thread_count(num_threads);

    const std::size_t n = input.size();
    output.resize(n);
    if (n == 0) return;

    const std::size_t padded_n = next_power_of_two(n);
    output.resize(padded_n);
    std::copy(input.begin(), input.end(), output.begin());
    std::fill(output.begin() + static_cast<std::ptrdiff_t>(n), output.end(), 0);

    omp_set_dynamic(0);

    #pragma omp parallel num_threads(num_threads)
    {
        // Up-sweep / reduce phase.
        // This reference version synchronizes after every tree level.
        for (std::size_t stride = 2; stride <= padded_n; stride <<= 1) {
            const std::size_t half = stride >> 1;
            const auto groups = static_cast<std::ptrdiff_t>(padded_n / stride);

            #pragma omp for schedule(static)
            for (std::ptrdiff_t group = 0; group < groups; ++group) {
                const std::size_t base = static_cast<std::size_t>(group) * stride;
                const std::size_t left = base + half - 1;
                const std::size_t right = base + stride - 1;
                output[right] += output[left];
            }

            if (stride == padded_n) {
                break;
            }
        }

        // Root becomes identity for exclusive scan.
        #pragma omp single
        {
            output[padded_n - 1] = 0;
        }

        // Down-sweep / distribute phase, also synchronized at every level.
        for (std::size_t stride = padded_n; stride > 1; stride >>= 1) {
            const std::size_t half = stride >> 1;
            const auto groups = static_cast<std::ptrdiff_t>(padded_n / stride);

            #pragma omp for schedule(static)
            for (std::ptrdiff_t group = 0; group < groups; ++group) {
                const std::size_t base = static_cast<std::size_t>(group) * stride;
                const std::size_t left = base + half - 1;
                const std::size_t right = base + stride - 1;

                const value_t temp = output[left];
                output[left] = output[right];
                output[right] += temp;
            }
        }
    }

    output.resize(n);
    if (scan_type == ScanType::Inclusive) {
        add_input_in_place(output, input, num_threads);
    }
}

std::vector<value_t> openmp_blelloch_exclusive_scan(
    const std::vector<value_t>& input,
    int num_threads
) {
    std::vector<value_t> output;
    openmp_blelloch_scan_into(input, output, num_threads, ScanType::Exclusive);
    return output;
}

std::vector<value_t> openmp_blelloch_inclusive_scan(
    const std::vector<value_t>& input,
    int num_threads
) {
    std::vector<value_t> output;
    openmp_blelloch_scan_into(input, output, num_threads, ScanType::Inclusive);
    return output;
}

std::vector<value_t> openmp_blelloch_scan(
    const std::vector<value_t>& input,
    int num_threads,
    ScanType scan_type
) {
    std::vector<value_t> output;
    openmp_blelloch_scan_into(input, output, num_threads, scan_type);
    return output;
}

void openmp_chunked_scan_into(
    const std::vector<value_t>& input,
    std::vector<value_t>& output,
    int num_threads,
    ScanType scan_type
) {
    ChunkedScanWorkspace workspace;
    openmp_chunked_scan_into(input, output, workspace, num_threads, scan_type);
}

void openmp_chunked_scan_into(
    const std::vector<value_t>& input,
    std::vector<value_t>& output,
    ChunkedScanWorkspace& workspace,
    int num_threads,
    ScanType scan_type
) {
    validate_thread_count(num_threads);

    const std::size_t n = input.size();
    output.resize(n);
    if (n == 0) return;

    omp_set_dynamic(0);

    workspace.resize(num_threads);

    #pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        const int actual_threads = omp_get_num_threads();
        const ChunkRange range = chunk_range(n, tid, actual_threads);

        value_t running_sum = 0;
        for (std::size_t i = range.start; i < range.end; ++i) {
            output[i] = running_sum;
            running_sum += input[i];
        }

        workspace.chunk_sums[static_cast<std::size_t>(tid)].value = running_sum;

        // Required: all chunk totals must be visible before one thread scans them.
        #pragma omp barrier

        #pragma omp single
        {
            value_t offset = 0;
            for (int i = 0; i < actual_threads; ++i) {
                const auto index = static_cast<std::size_t>(i);
                workspace.chunk_offsets[index].value = offset;
                offset += workspace.chunk_sums[index].value;
            }
        }

        // The implicit barrier after single ensures offsets are ready here.
        const value_t offset =
            workspace.chunk_offsets[static_cast<std::size_t>(tid)].value;
        const bool inclusive = scan_type == ScanType::Inclusive;

        if (inclusive) {
            for (std::size_t i = range.start; i < range.end; ++i) {
                output[i] += offset + input[i];
            }
        } else if (offset != 0) {
            for (std::size_t i = range.start; i < range.end; ++i) {
                output[i] += offset;
            }
        }
    }
}

std::vector<value_t> openmp_chunked_exclusive_scan(
    const std::vector<value_t>& input,
    int num_threads
) {
    std::vector<value_t> output;
    openmp_chunked_scan_into(input, output, num_threads, ScanType::Exclusive);
    return output;
}

std::vector<value_t> openmp_chunked_inclusive_scan(
    const std::vector<value_t>& input,
    int num_threads
) {
    std::vector<value_t> output;
    openmp_chunked_scan_into(input, output, num_threads, ScanType::Inclusive);
    return output;
}

std::vector<value_t> openmp_chunked_scan(
    const std::vector<value_t>& input,
    int num_threads,
    ScanType scan_type
) {
    std::vector<value_t> output;
    openmp_chunked_scan_into(input, output, num_threads, scan_type);
    return output;
}

bool verify_equal(
    const std::vector<value_t>& expected,
    const std::vector<value_t>& actual,
    std::size_t* mismatch_index
) {
    if (expected.size() != actual.size()) {
        if (mismatch_index) *mismatch_index = std::min(expected.size(), actual.size());
        return false;
    }

    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) {
            if (mismatch_index) *mismatch_index = i;
            return false;
        }
    }
    return true;
}

} // namespace scan
