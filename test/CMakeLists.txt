# NVSHMEM CXL Test CMakeLists.txt

cmake_minimum_required(VERSION 3.19)

# CXL P2P Test
add_executable(cxl_p2p_test cxl_p2p_test.cu)

target_include_directories(cxl_p2p_test
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_SOURCE_DIR}/src/host/transport/cxl
)

target_link_libraries(cxl_p2p_test
    PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
)

set_target_properties(cxl_p2p_test
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

if(NVSHMEM_TEST_RELEASE_PREFIX)
    install(
        TARGETS cxl_p2p_test
        RUNTIME DESTINATION ${NVSHMEM_TEST_RELEASE_PREFIX}
    )
endif()
