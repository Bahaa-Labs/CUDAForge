cmake_minimum_required(VERSION 3.22)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds are forbidden. Create a separate build directory (e.g., 'mkdir build && cd build').")
endif()

project(cudaforge 
    VERSION 0.1.0 
    DESCRIPTION "High-Performance LLM Inference & CUDA Kernel Engine"
    LANGUAGES CXX CUDA
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile commands for IDE integration" FORCE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CUDAFlags)
include(Dependencies)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type (Release, Debug, RelWithDebInfo)" FORCE)
endif()

message(STATUS " CUDAForge Engine Configuration Matrix")
message(STATUS " Build Type:            ${CMAKE_BUILD_TYPE}")
message(STATUS " C++ Standard:          C++${CMAKE_CXX_STANDARD}")
message(STATUS " CUDA Architecture:     ${CUDA_ARCH_TARGET}")
message(STATUS " CXX Compiler:          ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS " CUDA Compiler:         ${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}")

add_subdirectory(csrc)

if(NOT pybind11_DIR)
    execute_process(
        COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()

find_package(pybind11 REQUIRED)

pybind11_add_module(_C
    csrc/bindings/pybind_module.cpp
)

target_include_directories(_C PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_definitions(_C PRIVATE
    USE_CUDA
    WITH_CUDA
)

target_link_libraries(_C PRIVATE
    cudaforge_core
    CUDA::cudart
    CUDA::cublas
)

option(BUILD_TESTS "Build CUDAForge C++ Unit Tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp EXCLUDE_FROM_ALL)
endif()