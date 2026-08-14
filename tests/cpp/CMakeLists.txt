add_executable(cudaforge_cpp_tests
    test_block_allocator.cpp
    test_buffer_pool.cpp
    test_paged_kv_cache.cpp
)

target_include_directories(cudaforge_cpp_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/csrc/include
    ${CUDA_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(cudaforge_cpp_tests PRIVATE
    cudaforge_core
    gtest
    gtest_main
    CUDA::cudart
    spdlog::spdlog
)

target_compile_features(cudaforge_cpp_tests PRIVATE cxx_std_20)

include(GoogleTest)
gtest_discover_tests(cudaforge_cpp_tests)