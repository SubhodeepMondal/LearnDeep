set(LIB_INSTALL_DIR ${CMAKE_SOURCE_DIR}/../lib)
link_directories(${LIB_INSTALL_DIR})

add_executable(deeplearingLA_test graph_test.cpp)
set(LIBS framework graph kernel las )
target_include_directories(deeplearingLA_test PRIVATE ${CMAKE_SOURCE_DIR}/api)

target_link_directories( deeplearingLA_test PRIVATE ${LIB_INSTALL_DIR})
target_link_libraries(deeplearingLA_test ${LIBS} OpenMP::OpenMP_CXX)