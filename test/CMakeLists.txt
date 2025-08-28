
# Unit test suite for deeplearingLA

add_executable(deeplearingLA_test unit/main.cpp
    unit/LinearAlgebraFixtures.unit.hpp
    unit/LinearAlgebraFixtures.unit.cpp
    unit/tensor.unit.cpp
    unit/graph.unit.cpp
    unit/addition.unit.cpp
    unit/multiplication.unit.cpp
    unit/matmul.unit.cpp
    unit/reducesum.unit.cpp
    unit/gemm.unit.cpp
    unit/scale.unit.cpp
    unit/power.unit.cpp
    unit/relu.unit.cpp
    unit/mean.unit.cpp
    unit/sub.unit.cpp
    unit/sqrt.unit.cpp
)

target_include_directories(deeplearingLA_test PRIVATE ${CMAKE_SOURCE_DIR}/api)

target_link_libraries(deeplearingLA_test PRIVATE tensor_api thirdparty::gtest_main)

# Benchmarking suite for deeplearingLA
add_executable(benchmark_test benchmark/main.benchmark.cpp
    benchmark/addition.benchmark.cpp
    benchmark/multiplication.benchmark.cpp
    benchmark/matmul.benchmark.cpp
    # benchmark/mean.benchmark.cpp
    # benchmark/reducesum.benchmark.cpp
    # benchmark/scale.benchmark.cpp
    benchmark/power.benchmark.cpp
    # benchmark/relu.benchmark.cpp
    # benchmark/sub.benchmark.cpp
    # benchmark/sqrt.benchmark.cpp
)

target_include_directories(benchmark_test PRIVATE ${CMAKE_SOURCE_DIR}/api
    ${CMAKE_SOURCE_DIR}/core)

target_link_libraries(benchmark_test PRIVATE
    tensor_api
    thirdparty::benchmark
    absl::log
    absl::log_initialize
    absl::flags_parse
)
