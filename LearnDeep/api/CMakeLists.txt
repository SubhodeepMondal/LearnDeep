
enable_language(CXX)

add_library(layer_api SHARED
    layer_api.cpp
    layer_api_helpers.hpp
)

target_include_directories(layer_api PUBLIC ${CMAKE_SOURCE_DIR})

target_link_libraries(layer_api
    PUBLIC
    core
    layer
    absl::log
    absl::log_initialize
    absl::flags_parse
)

add_library(tensor_api SHARED
    tensor.h
    tensor.cpp
)

target_include_directories(tensor_api PUBLIC ${CMAKE_SOURCE_DIR})


target_link_libraries(tensor_api
    PUBLIC
    core
    layer
    layer_api
    loss
    model
    optimizer
    callback
    absl::log
    absl::log_initialize
    absl::flags_parse
)
