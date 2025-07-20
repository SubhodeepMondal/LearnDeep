add_library(tensor_api SHARED
    tensor.cpp
)

target_include_directories(tensor_api PUBLIC ${CMAKE_SOURCE_DIR}/core)
target_include_directories(kernel PUBLIC ${CMAKE_SOURCE_DIR}/core/)