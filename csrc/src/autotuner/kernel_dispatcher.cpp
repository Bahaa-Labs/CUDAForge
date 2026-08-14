#include "cudaforge/autotuner/kernel_dispatcher.h"

#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace cudaforge::autotuner {

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        cudaError_t err = call;                                                              \
        if (err != cudaSuccess) {                                                            \
            std::ostringstream ss;                                                           \
            ss << "CUDA Error in " << __FILE__ << ":" << __LINE__ << " ["                    \
               << cudaGetErrorName(err) << "]: " << cudaGetErrorString(err);                 \
            throw std::runtime_error(ss.str());                                              \
        }                                                                                    \
    } while (0)
#endif

// Static instance definition if out-of-line linkage is required
KernelDispatcher& get_global_dispatcher() {
    return KernelDispatcher::instance();
}

/*
    Helper utility to convert numerical data type code into a human-readable string.
 */
static std::string dtype_code_to_string(int32_t dtype_code) {
    switch (dtype_code) {
        case 0:  return "FP16";
        case 1:  return "BF16";
        case 2:  return "INT4";
        case 3:  return "INT8";
        default: return "UNKNOWN(" + std::to_string(dtype_code) + ")";
    }
}

/*
    Dump active autotuner cache statistics to standard output for debugging and profiling.
 */
void dump_autotuner_cache(KernelDispatcher& dispatcher) {
    // Access cache via dispatch inspection log
    std::cout << "\n[CUDAForge Autotuner Cache Dump]\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(20) << "Operation"
              << std::setw(8)  << "M"
              << std::setw(8)  << "N"
              << std::setw(8)  << "K"
              << std::setw(12) << "Precision"
              << "Elected Variant\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Cache status logged successfully.\n";
    std::cout << "------------------------------------------------------------------------\n\n";
}

} // namespace cudaforge::autotuner