
add_executable(deeplearingLA_test unit/main.cpp
    unit/LinearAlgebraFixtures.unit.hpp
    unit/LinearAlgebraFixtures.unit.cpp
    unit/tensor.unit.cpp
    unit/graph.unit.cpp
    unit/addition.unit.cpp
    unit/multiplication.unit.cpp
    unit/matmul.unit.cpp
)
# set(LIBS fcore tensor_api)
target_include_directories(deeplearingLA_test PRIVATE ${CMAKE_SOURCE_DIR}/api
    ${CMAKE_SOURCE_DIR}/thirdparty/googletest/include)

target_link_libraries(deeplearingLA_test PRIVATE core gpu tensor_api gtest gtest_main OpenMP::OpenMP_CXX)
