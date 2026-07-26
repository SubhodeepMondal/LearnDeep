# CMakeLists.txt for the layer library

set(LAYER_RESOURCES
    dense.hpp
    dense.cpp
    layer_graph.cpp
    layer_graph.hpp
    layers.cpp
    layers.hpp
)

add_library(layer SHARED ${LAYER_RESOURCES}
    relu.cpp)
target_include_directories(layer PUBLIC ${CMAKE_SOURCE_DIR})

target_link_libraries(layer PUBLIC core)