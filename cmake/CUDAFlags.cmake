include(CheckLanguage)
check_language(CUDA)

if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA compiler (nvcc) not found. Ensure CUDA Toolkit 12.x/13.x is installed and in PATH.")
endif()

set(CUDA_ARCH_TARGET "86" CACHE STRING "Target CUDA Compute Capability (e.g., 86 for Ampere)")

set(CUDA_ARCH_FLAGS "-gencode=arch=compute_${CUDA_ARCH_TARGET},code=sm_${CUDA_ARCH_TARGET}")
list(APPEND CUDA_ARCH_FLAGS "-gencode=arch=compute_${CUDA_ARCH_TARGET},code=compute_${CUDA_ARCH_TARGET}")

set(NVCC_HOST_CXX_FLAGS
    "-Xcompiler=-fPIC"
    "-Xcompiler=-Wall"
    "-Xcompiler=-Wextra"
    "-Xcompiler=-Wno-strict-aliasing"
    "-Xcompiler=-Wno-unused-parameter"
)

set(NVCC_PERF_FLAGS
    "-O3"
    "--use_fast_math"                   
    "--expt-relaxed-constexpr"          
    "--expt-extended-lambda"           
    "--threads=4"                       
    "-Xptxas=-O3"                       
    "-Xptxas=-v"                        
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND NVCC_PERF_FLAGS "-G" "-lineinfo")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    list(APPEND NVCC_PERF_FLAGS "-lineinfo")
endif()

string(JOIN " " NVCC_FLAGS_STRING
    ${CUDA_ARCH_FLAGS}
    ${NVCC_HOST_CXX_FLAGS}
    ${NVCC_PERF_FLAGS}
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${NVCC_FLAGS_STRING}")

find_package(CUDAToolkit REQUIRED)

message(STATUS "CUDA Toolkit Found: Version ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Include Directories: ${CUDAToolkit_INCLUDE_DIRS}")