cmake_minimum_required(VERSION 3.15)
project(TENSOR_CPP_PROJECT)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenMP REQUIRED)

set(LIB_INSTALL_DIR ${CMAKE_SOURCE_DIR}/../lib)
link_directories(${LIB_INSTALL_DIR})

add_executable(${PROJECT_NAME} graph_test.cpp)
set(LIBS framework graph kernel las )

target_link_directories( ${PROJECT_NAME} PRIVATE ${LIB_INSTALL_DIR})
target_link_libraries(${PROJECT_NAME} ${LIBS} OpenMP::OpenMP_CXX)