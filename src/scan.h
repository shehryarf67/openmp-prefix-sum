#ifndef SCAN_H
#define SCAN_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace scan {

using value_t = std::uint64_t;

enum class ScanType {
    Exclusive,
    Inclusive
};

std::size_t next_power_of_two(std::size_t n);

// Sequential reference implementations.
void sequential_scan_into(
    const std::vector<value_t>& input,
    std::vector<value_t>& output,
    ScanType scan_type
);
std::vector<value_t> sequential_exclusive_scan(const std::vector<value_t>& input);
std::vector<value_t> sequential_inclusive_scan(const std::vector<value_t>& input);
std::vector<value_t> sequential_scan(
    const std::vector<value_t>& input,
    ScanType scan_type
);

// Direct OpenMP Blelloch scan over a padded power-of-two tree.
std::vector<value_t> openmp_blelloch_exclusive_scan(
    const std::vector<value_t>& input,
    int num_threads
);
std::vector<value_t> openmp_blelloch_inclusive_scan(
    const std::vector<value_t>& input,
    int num_threads
);
std::vector<value_t> openmp_blelloch_scan(
    const std::vector<value_t>& input,
    int num_threads,
    ScanType scan_type
);

// Practical n > p OpenMP version:
// local chunk scans, scan chunk sums, then add chunk offsets.
void openmp_chunked_scan_into(
    const std::vector<value_t>& input,
    std::vector<value_t>& output,
    int num_threads,
    ScanType scan_type
);
std::vector<value_t> openmp_chunked_exclusive_scan(
    const std::vector<value_t>& input,
    int num_threads
);
std::vector<value_t> openmp_chunked_inclusive_scan(
    const std::vector<value_t>& input,
    int num_threads
);
std::vector<value_t> openmp_chunked_scan(
    const std::vector<value_t>& input,
    int num_threads,
    ScanType scan_type
);

bool verify_equal(
    const std::vector<value_t>& expected,
    const std::vector<value_t>& actual,
    std::size_t* mismatch_index = nullptr
);

} // namespace scan

#endif
