set(LIB_INSTALL_DIR ${CMAKE_SOURCE_DIR}/../lib)
link_directories(${LIB_INSTALL_DIR})


add_executable(deeplearingLA_test unit/main.cpp
    unit/LinearAlgebraFixtures.unit.cpp
    unit/addition.unit.cpp
    unit/multiplication.unit.cpp
)
set(LIBS framework graph kernel las gpu gtest tensor_api)
target_include_directories(deeplearingLA_test PRIVATE ${CMAKE_SOURCE_DIR}/api
    ${CMAKE_SOURCE_DIR}/thirdparty/googletest/include)

target_link_directories(deeplearingLA_test PRIVATE ${LIB_INSTALL_DIR}
    ${CMAKE_SOURCE_DIR}/thirdparty/googletest/include)
target_link_libraries(deeplearingLA_test ${LIBS} OpenMP::OpenMP_CXX)
