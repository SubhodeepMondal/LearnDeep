add_library(tensor_api SHARED
    tensor.h
    tensor.cpp
)

target_include_directories(tensor_api PUBLIC ${CMAKE_SOURCE_DIR}/core)
# target_include_directories(kernel PUBLIC ${CMAKE_SOURCE_DIR}/core/)
# target_link_libraries(tensor_api framework graph kernel)


target_link_libraries(tensor_api
    PRIVATE
    core
    gpu 
    OpenMP::OpenMP_CXX
)