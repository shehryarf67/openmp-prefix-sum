#include "scan.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <omp.h>

namespace scan {

namespace {

struct alignas(64) PaddedSum {
    value_t value = 0;
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

int get_actual_thread_count(int requested_threads) {
    int actual_threads = 1;

    #pragma omp parallel num_threads(requested_threads)
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }

    return actual_threads;
}

} // namespace

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
    std::vector<value_t> output(input.size(), 0);
    value_t running_sum = 0;

    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = running_sum;
        running_sum += input[i];
    }

    return output;
}

std::vector<value_t> sequential_inclusive_scan(const std::vector<value_t>& input) {
    std::vector<value_t> output(input.size(), 0);
    value_t running_sum = 0;

    for (std::size_t i = 0; i < input.size(); ++i) {
        running_sum += input[i];
        output[i] = running_sum;
    }

    return output;
}

std::vector<value_t> sequential_scan(
    const std::vector<value_t>& input,
    ScanType scan_type
) {
    if (scan_type == ScanType::Exclusive) {
        return sequential_exclusive_scan(input);
    }
    return sequential_inclusive_scan(input);
}

std::vector<value_t> openmp_blelloch_exclusive_scan(
    const std::vector<value_t>& input,
    int num_threads
) {
    validate_thread_count(num_threads);

    const std::size_t n = input.size();
    if (n == 0) return {};

    const std::size_t padded_n = next_power_of_two(n);
    std::vector<value_t> tree(padded_n, 0);
    std::copy(input.begin(), input.end(), tree.begin());

    omp_set_dynamic(0);

    #pragma omp parallel num_threads(num_threads)
    {
        // Up-sweep / reduce phase.
        for (std::size_t stride = 2; stride <= padded_n; stride <<= 1) {
            const std::size_t half = stride >> 1;
            const auto groups = static_cast<std::ptrdiff_t>(padded_n / stride);

            #pragma omp for schedule(static)
            for (std::ptrdiff_t group = 0; group < groups; ++group) {
                const std::size_t base = static_cast<std::size_t>(group) * stride;
                const std::size_t left = base + half - 1;
                const std::size_t right = base + stride - 1;
                tree[right] += tree[left];
            }

            if (stride == padded_n) {
                break;
            }
        }

        // Root becomes identity for exclusive scan.
        #pragma omp single
        {
            tree[padded_n - 1] = 0;
        }

        // Down-sweep / distribute phase.
        for (std::size_t stride = padded_n; stride > 1; stride >>= 1) {
            const std::size_t half = stride >> 1;
            const auto groups = static_cast<std::ptrdiff_t>(padded_n / stride);

            #pragma omp for schedule(static)
            for (std::ptrdiff_t group = 0; group < groups; ++group) {
                const std::size_t base = static_cast<std::size_t>(group) * stride;
                const std::size_t left = base + half - 1;
                const std::size_t right = base + stride - 1;

                const value_t temp = tree[left];
                tree[left] = tree[right];
                tree[right] += temp;
            }
        }
    }

    tree.resize(n);
    return tree;
}

std::vector<value_t> openmp_blelloch_inclusive_scan(
    const std::vector<value_t>& input,
    int num_threads
) {
    std::vector<value_t> output = openmp_blelloch_exclusive_scan(input, num_threads);
    add_input_in_place(output, input, num_threads);
    return output;
}

std::vector<value_t> openmp_blelloch_scan(
    const std::vector<value_t>& input,
    int num_threads,
    ScanType scan_type
) {
    if (scan_type == ScanType::Exclusive) {
        return openmp_blelloch_exclusive_scan(input, num_threads);
    }
    return openmp_blelloch_inclusive_scan(input, num_threads);
}

std::vector<value_t> openmp_chunked_exclusive_scan(
    const std::vector<value_t>& input,
    int num_threads
) {
    validate_thread_count(num_threads);

    const std::size_t n = input.size();
    if (n == 0) return {};

    omp_set_dynamic(0);

    const int actual_threads = get_actual_thread_count(num_threads);

    std::vector<value_t> output(n, 0);
    std::vector<PaddedSum> padded_chunk_sums(static_cast<std::size_t>(actual_threads));

    // Local exclusive scan inside each chunk.
    #pragma omp parallel num_threads(actual_threads)
    {
        const int tid = omp_get_thread_num();
        const std::size_t start = (n * static_cast<std::size_t>(tid)) / static_cast<std::size_t>(actual_threads);
        const std::size_t end = (n * static_cast<std::size_t>(tid + 1)) / static_cast<std::size_t>(actual_threads);

        value_t running_sum = 0;
        for (std::size_t i = start; i < end; ++i) {
            output[i] = running_sum;
            running_sum += input[i];
        }
        padded_chunk_sums[static_cast<std::size_t>(tid)].value = running_sum;
    }

    std::vector<value_t> chunk_sums(static_cast<std::size_t>(actual_threads), 0);
    for (int i = 0; i < actual_threads; ++i) {
        chunk_sums[static_cast<std::size_t>(i)] = padded_chunk_sums[static_cast<std::size_t>(i)].value;
    }

    // Scan chunk totals to compute each thread's global offset.
    std::vector<value_t> chunk_offsets = openmp_blelloch_exclusive_scan(chunk_sums, actual_threads);

    // Add each chunk's offset to its local scan result.
    #pragma omp parallel num_threads(actual_threads)
    {
        const int tid = omp_get_thread_num();
        const std::size_t start = (n * static_cast<std::size_t>(tid)) / static_cast<std::size_t>(actual_threads);
        const std::size_t end = (n * static_cast<std::size_t>(tid + 1)) / static_cast<std::size_t>(actual_threads);
        const value_t offset = chunk_offsets[static_cast<std::size_t>(tid)];

        for (std::size_t i = start; i < end; ++i) {
            output[i] += offset;
        }
    }

    return output;
}

std::vector<value_t> openmp_chunked_inclusive_scan(
    const std::vector<value_t>& input,
    int num_threads
) {
    validate_thread_count(num_threads);

    const std::size_t n = input.size();
    if (n == 0) return {};

    omp_set_dynamic(0);

    const int actual_threads = get_actual_thread_count(num_threads);

    std::vector<value_t> output(n, 0);
    std::vector<PaddedSum> padded_chunk_sums(static_cast<std::size_t>(actual_threads));

    // Local inclusive scan inside each chunk.
    #pragma omp parallel num_threads(actual_threads)
    {
        const int tid = omp_get_thread_num();
        const std::size_t start = (n * static_cast<std::size_t>(tid)) / static_cast<std::size_t>(actual_threads);
        const std::size_t end = (n * static_cast<std::size_t>(tid + 1)) / static_cast<std::size_t>(actual_threads);

        value_t running_sum = 0;
        for (std::size_t i = start; i < end; ++i) {
            running_sum += input[i];
            output[i] = running_sum;
        }
        padded_chunk_sums[static_cast<std::size_t>(tid)].value = running_sum;
    }

    std::vector<value_t> chunk_sums(static_cast<std::size_t>(actual_threads), 0);
    for (int i = 0; i < actual_threads; ++i) {
        chunk_sums[static_cast<std::size_t>(i)] = padded_chunk_sums[static_cast<std::size_t>(i)].value;
    }

    std::vector<value_t> chunk_offsets = openmp_blelloch_exclusive_scan(chunk_sums, actual_threads);

    #pragma omp parallel num_threads(actual_threads)
    {
        const int tid = omp_get_thread_num();
        const std::size_t start = (n * static_cast<std::size_t>(tid)) / static_cast<std::size_t>(actual_threads);
        const std::size_t end = (n * static_cast<std::size_t>(tid + 1)) / static_cast<std::size_t>(actual_threads);
        const value_t offset = chunk_offsets[static_cast<std::size_t>(tid)];

        for (std::size_t i = start; i < end; ++i) {
            output[i] += offset;
        }
    }

    return output;
}

std::vector<value_t> openmp_chunked_scan(
    const std::vector<value_t>& input,
    int num_threads,
    ScanType scan_type
) {
    if (scan_type == ScanType::Exclusive) {
        return openmp_chunked_exclusive_scan(input, num_threads);
    }
    return openmp_chunked_inclusive_scan(input, num_threads);
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
