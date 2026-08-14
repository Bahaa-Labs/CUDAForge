set(CMAKE_CUDA_ARCHITECTURES 86)

add_library(cudaforge_core SHARED
    src/memory/block_allocator.cpp
    src/memory/buffer_pool.cpp
    src/memory/paged_kv_cache.cpp
    src/scheduler/continuous_batcher.cpp
    src/autotuner/kernel_dispatcher.cpp
    kernels/cuda/flash_attn_v2.cu
    kernels/cuda/fused_rmsnorm.cu
    kernels/cuda/int4_gemm_dequant.cu
)

set_target_properties(cudaforge_core PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    OUTPUT_NAME "cudaforge_core"
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CXX_STANDARD 20
    CUDA_STANDARD 17
)

target_include_directories(cudaforge_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/kernels>
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_definitions(cudaforge_core PRIVATE
    USE_CUDA
    WITH_CUDA
    CUDA_HAS_FP16=1
    CUDA_HAS_BF16=1
)

target_link_libraries(cudaforge_core PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    pybind11::module
    spdlog::spdlog
)