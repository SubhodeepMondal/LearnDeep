
# Unit test suite for deeplearingLA

add_executable(unit_test
    unit/main.cpp
    unit/addition.unit.cpp
    unit/gemm.unit.cpp
    unit/graph.unit.cpp
    unit/LinearAlgebraFixtures.unit.hpp
    unit/LinearAlgebraFixtures.unit.cpp
    unit/matmul.unit.cpp
    unit/mean.unit.cpp
    unit/multiplication.unit.cpp
    unit/power.unit.cpp
    unit/reducesum.unit.cpp
    unit/relu.unit.cpp
    unit/scale.unit.cpp
    unit/sigmoid.unit.cpp
    # unit/softmax.unit.cpp
    unit/sqrt.unit.cpp
    unit/sub.unit.cpp
    unit/tensor.unit.cpp
)

target_include_directories(unit_test PRIVATE ${CMAKE_SOURCE_DIR}/api)

target_link_libraries(unit_test PRIVATE
    tensor_api
    GTest::gtest_main
    absl::log
    absl::log_initialize
    absl::flags_parse
)

# Benchmarking suite for deeplearingLA
add_executable(benchmark_test
    benchmark/addition.benchmark.cpp
    benchmark/main.benchmark.cpp
    benchmark/matmul.benchmark.cpp
    benchmark/mean.benchmark.cpp
    benchmark/multiplication.benchmark.cpp
    benchmark/power.benchmark.cpp
    benchmark/reducesum.benchmark.cpp
    benchmark/scale.benchmark.cpp
    # benchmark/sigmoid.benchmark.cpp
    # benchmark/relu.benchmark.cpp
    # benchmark/sub.benchmark.cpp
    benchmark/sqrt.benchmark.cpp
)

target_include_directories(benchmark_test PRIVATE ${CMAKE_SOURCE_DIR}/api
    ${CMAKE_SOURCE_DIR}/core)

target_link_libraries(benchmark_test PRIVATE
    tensor_api
    benchmark::benchmark
    absl::log
    absl::log_initialize
    absl::flags_parse
)

add_test(NAME unit_test COMMAND unit_test)
include(GoogleTest)            # provides gtest_discover_tests
gtest_discover_tests(unit_test WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_test(NAME benchmark_test COMMAND benchmark_test)
# set_tests_properties(benchmark_test PROPERTIES TIMEOUT 300) # Set a timeout of 300 seconds (5 minutes)

