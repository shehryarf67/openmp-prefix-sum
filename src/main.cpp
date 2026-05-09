#include "scan.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <omp.h>

namespace {

struct Options {
    std::size_t n = (1ULL << 22) + 3;
    int threads = 4;
    int repeats = 3;
    std::string mode = "both"; // sequential, direct, chunked, both
    scan::ScanType scan_type = scan::ScanType::Exclusive;
    bool csv = false;
    bool run_tests = false;
};

volatile scan::value_t benchmark_sink = 0;

std::string scan_type_name(scan::ScanType scan_type) {
    return scan_type == scan::ScanType::Exclusive ? "exclusive" : "inclusive";
}

scan::ScanType parse_scan_type(const std::string& value) {
    if (value == "exclusive") {
        return scan::ScanType::Exclusive;
    }
    if (value == "inclusive") {
        return scan::ScanType::Inclusive;
    }
    throw std::invalid_argument("scan type must be exclusive or inclusive");
}

bool valid_mode(const std::string& mode) {
    return mode == "sequential" || mode == "direct" || mode == "chunked" || mode == "both";
}

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "Options:\n"
        << "  --n <N>              Input size (default: 2^22 + 3)\n"
        << "  --threads <T>        OpenMP thread count (default: 4)\n"
        << "  --repeats <R>        Timed repetitions (default: 3)\n"
        << "  --mode <M>           sequential | direct | chunked | both (default: both)\n"
        << "  --scan-type <S>      exclusive | inclusive (default: exclusive)\n"
        << "  --csv                Print compact CSV row\n"
        << "  --test               Run correctness tests and exit\n"
        << "  --help               Show this message\n";
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--n" && i + 1 < argc) {
            options.n = static_cast<std::size_t>(std::stoull(argv[++i]));
        } else if (arg == "--threads" && i + 1 < argc) {
            options.threads = std::stoi(argv[++i]);
        } else if (arg == "--repeats" && i + 1 < argc) {
            options.repeats = std::stoi(argv[++i]);
        } else if (arg == "--mode" && i + 1 < argc) {
            options.mode = argv[++i];
        } else if (arg == "--scan-type" && i + 1 < argc) {
            options.scan_type = parse_scan_type(argv[++i]);
        } else if (arg == "--csv") {
            options.csv = true;
        } else if (arg == "--test") {
            options.run_tests = true;
        } else {
            std::cerr << "Unknown or incomplete option: " << arg << "\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }

    if (options.threads < 1) options.threads = 1;
    if (options.repeats < 1) options.repeats = 1;
    if (!valid_mode(options.mode)) {
        throw std::invalid_argument("mode must be sequential, direct, chunked, or both");
    }
    return options;
}

std::vector<scan::value_t> make_input(std::size_t n) {
    std::vector<scan::value_t> input(n);
    for (std::size_t i = 0; i < n; ++i) {
        // Small bounded values keep sums readable while still testing correctness.
        input[i] = static_cast<scan::value_t>(((i * 13) + 7) % 29);
    }
    return input;
}

template <typename Func>
double benchmark_ms(Func&& func, int repeats) {
    double total_ms = 0.0;
    for (int r = 0; r < repeats; ++r) {
        const auto start = std::chrono::high_resolution_clock::now();
        const scan::value_t guard = func();
        const auto end = std::chrono::high_resolution_clock::now();
        benchmark_sink = static_cast<scan::value_t>(benchmark_sink ^ guard);
        total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }
    return total_ms / static_cast<double>(repeats);
}

scan::value_t output_guard(const std::vector<scan::value_t>& output) {
    if (output.empty()) {
        return 0;
    }
    return output.front() ^ output.back() ^ static_cast<scan::value_t>(output.size());
}

struct TestCase {
    std::string name;
    std::vector<scan::value_t> input;
};

bool run_correctness_tests() {
    std::vector<TestCase> cases = {
        {"empty", {}},
        {"single_nonzero", {42}},
        {"single_zero", {0}},
        {"small_power_of_two", {3, 1, 7, 0, 4, 1, 6, 3}},
        {"small_non_power_of_two", {5, 0, 2, 9, 1, 4, 8}},
        {"sixteen_items", make_input(16)},
        {"seventeen_items", make_input(17)},
        {"large_non_power_of_two", make_input(100003)}
    };

    std::vector<int> thread_counts = {1, 2, 3, 4, 8};
    const int max_threads = omp_get_max_threads();
    if (max_threads > 0) {
        thread_counts.push_back(max_threads);
    }
    std::sort(thread_counts.begin(), thread_counts.end());
    thread_counts.erase(
        std::unique(thread_counts.begin(), thread_counts.end()),
        thread_counts.end()
    );

    std::size_t checks = 0;
    std::size_t failures = 0;

    for (const auto scan_type : {scan::ScanType::Exclusive, scan::ScanType::Inclusive}) {
        for (const TestCase& test_case : cases) {
            const std::vector<scan::value_t> expected =
                scan::sequential_scan(test_case.input, scan_type);

            for (int threads : thread_counts) {
                const std::vector<std::pair<std::string, std::vector<scan::value_t>>> results = {
                    {"direct", scan::openmp_blelloch_scan(test_case.input, threads, scan_type)},
                    {"chunked", scan::openmp_chunked_scan(test_case.input, threads, scan_type)}
                };

                for (const auto& result : results) {
                    ++checks;
                    std::size_t mismatch = 0;
                    const bool ok = scan::verify_equal(expected, result.second, &mismatch);
                    if (!ok) {
                        ++failures;
                        std::cout << "[FAIL] scan_type=" << scan_type_name(scan_type)
                                  << " case=" << test_case.name
                                  << " n=" << test_case.input.size()
                                  << " threads=" << threads
                                  << " mode=" << result.first
                                  << " mismatch_index=" << mismatch;
                        if (mismatch < expected.size() && mismatch < result.second.size()) {
                            std::cout << " expected=" << expected[mismatch]
                                      << " actual=" << result.second[mismatch];
                        }
                        std::cout << '\n';
                    }
                }
            }
        }
    }

    if (failures == 0) {
        std::cout << "Correctness tests: PASS (" << checks << " checks)\n";
    } else {
        std::cout << "Correctness tests: FAIL (" << failures << " of "
                  << checks << " checks failed)\n";
    }

    return failures == 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);

        if (options.run_tests) {
            return run_correctness_tests() ? 0 : 1;
        }

        const std::vector<scan::value_t> input = make_input(options.n);

        std::vector<scan::value_t> seq_output;
        const double seq_ms = benchmark_ms([&]() -> scan::value_t {
            seq_output = scan::sequential_scan(input, options.scan_type);
            return output_guard(seq_output);
        }, options.repeats);

        if (options.csv) {
            std::cout
                << "n,threads,scan_type,mode,sequential_ms,"
                << "parallel_ms,speedup,efficiency,status\n";
        }

        if (options.mode == "sequential") {
            if (options.csv) {
                std::cout << options.n << ',' << options.threads << ','
                          << scan_type_name(options.scan_type) << ",sequential,"
                          << std::fixed << std::setprecision(4)
                          << seq_ms << ",,,,OK\n";
            } else {
                std::cout << "Mode: sequential\n";
                std::cout << "Scan type: " << scan_type_name(options.scan_type) << "\n";
                std::cout << "Input size: " << options.n << "\n";
                std::cout << std::fixed << std::setprecision(4);
                std::cout << "Sequential average time (ms): " << seq_ms << "\n";
                std::cout << "Correctness: PASS (reference run)\n\n";
            }
            return 0;
        }

        auto run_parallel = [&](const std::string& label) {
            std::vector<scan::value_t> par_output;
            double par_ms = 0.0;

            if (label == "direct") {
                par_ms = benchmark_ms([&]() -> scan::value_t {
                    par_output = scan::openmp_blelloch_scan(
                        input,
                        options.threads,
                        options.scan_type
                    );
                    return output_guard(par_output);
                }, options.repeats);
            } else if (label == "chunked") {
                par_ms = benchmark_ms([&]() -> scan::value_t {
                    par_output = scan::openmp_chunked_scan(
                        input,
                        options.threads,
                        options.scan_type
                    );
                    return output_guard(par_output);
                }, options.repeats);
            } else {
                throw std::invalid_argument("unknown parallel mode");
            }

            std::size_t mismatch = 0;
            const bool ok = scan::verify_equal(seq_output, par_output, &mismatch);
            const double speedup = par_ms > 0.0 ? seq_ms / par_ms : 0.0;
            const double efficiency = speedup / static_cast<double>(options.threads);

            if (options.csv) {
                std::cout << options.n << ',' << options.threads << ','
                          << scan_type_name(options.scan_type) << ',' << label << ','
                          << std::fixed << std::setprecision(4)
                          << seq_ms << ',' << par_ms << ',' << speedup << ','
                          << efficiency << ',' << (ok ? "OK" : "FAIL") << '\n';
            } else {
                std::cout << "Mode: " << label << "\n";
                std::cout << "Scan type: " << scan_type_name(options.scan_type) << "\n";
                std::cout << "Input size: " << options.n << "\n";
                std::cout << "Threads: " << options.threads << "\n";
                std::cout << "Repeats: " << options.repeats << "\n";
                std::cout << std::fixed << std::setprecision(4);
                std::cout << "Sequential average time (ms): " << seq_ms << "\n";
                std::cout << "Parallel average time (ms):   " << par_ms << "\n";
                std::cout << "Speedup:                      " << speedup << "\n";
                std::cout << "Efficiency:                   " << efficiency << "\n";
                std::cout << "Correctness:                  " << (ok ? "PASS" : "FAIL") << "\n";
                if (!ok) {
                    std::cout << "Mismatch index:               " << mismatch << "\n";
                    std::cout << "Expected:                     " << seq_output[mismatch] << "\n";
                    std::cout << "Actual:                       " << par_output[mismatch] << "\n";
                }
                std::cout << "\n";
            }
        };

        if (options.mode == "direct" || options.mode == "both") {
            run_parallel("direct");
        }
        if (options.mode == "chunked" || options.mode == "both") {
            run_parallel("chunked");
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
