cmake_minimum_required(VERSION 3.15)
project(kernel)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_library(${PROJECT_NAME} opskernel.cpp add.cpp matmul.cpp mul.cpp power.cpp reducesum.cpp scale.cpp)
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/../../)
install(TARGETS ${PROJECT_NAME}  
        LIBRARY DESTINATION lib)