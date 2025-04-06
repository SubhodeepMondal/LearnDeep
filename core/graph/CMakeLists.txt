cmake_minimum_required(VERSION 3.15)
project(graph)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_library(${PROJECT_NAME} graph.cpp)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/../../)
install(TARGETS ${PROJECT_NAME}  
        LIBRARY DESTINATION lib)