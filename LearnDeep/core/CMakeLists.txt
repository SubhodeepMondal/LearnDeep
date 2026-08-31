# CMakeLists.txt for the core library

#creating library for kernel operations

set(COMBINED_RESORCES
    ops/add.cpp
    ops/div.cpp
    ops/matmul.cpp
    ops/mul.cpp
    ops/opskernel.cpp
    ops/opskernel.h
    ops/greaterthanzero.cpp
    ops/power.cpp
    ops/reducesum.cpp
    ops/relu.cpp
    ops/scale.cpp
    ops/softmax.cpp
    ops/sigmoid.cpp
    ops/sub.cpp
    ops/sqrt.cpp
    ops/transpose.cpp
    ops/kernelmanager.cpp
    ops/kernelmanager.h

    graph/graph_context.hpp
    graph/graph_framework.hpp
    graph/graph_framework.cpp
    graph/graph_manager.hpp
    graph/graph_manager.cpp
    graph/graph_node.hpp
    graph/graph_node.cpp

    framework/log_config.hpp
    framework/MathLibrary.h
    framework/MathLibrary.cpp
    framework/NDynamicArray.h
    framework/NDynamicArray.cpp


    LAS/CPULibrary.h
    LAS/CPULibrary.cpp
    LAS/avx2_micro_kernels.h
    LAS/avx2_micro_kernels.cpp

    utility/initializers.hpp
    utility/initializers.cpp
)

add_library(core SHARED ${COMBINED_RESORCES})
target_link_libraries(core PUBLIC OpenMP::OpenMP_CXX absl::log)
target_include_directories(core PUBLIC ${CMAKE_SOURCE_DIR}/LearnDeep)
target_include_directories(core SYSTEM PRIVATE ${CMAKE_BINARY_DIR}/_deps/abseil-src)

if(ENABLE_CUDA)
enable_language(CUDA)
endif()

if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
    message(STATUS "CUDA found: enabling CUDA support")
    enable_language(CUDA)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    add_compile_definitions(-DENABLE_CUDA)

    add_library(gpu SHARED
        LAS/gpu_interface.cuh
        LAS/gpu_interface.cu
        LAS/gpu_micro_kernels.cuh
        LAS/gpu_micro_kernels.cu
    )

    # Enable separate compilation for nested kernel launches
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

    set_target_properties(gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # Suppress deprecated arch warning
    target_compile_options(gpu PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--relocatable-device-code=true>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>")

    target_include_directories(gpu PRIVATE ${CMAKE_SOURCE_DIR}/LearnDeep/
        /usr/local/cuda/include)

    target_include_directories(gpu SYSTEM PRIVATE ${CMAKE_BINARY_DIR}/_deps/abseil-src)

    target_link_libraries(gpu PUBLIC
        absl::log
        absl::log_initialize
        absl::flags_parse)

    # Or set architecture explicitly
    set(CMAKE_CUDA_ARCHITECTURES 90) # or 80, 86, etc.


    # Link core with gpu (so test only needs to link core)
    target_link_libraries(core PUBLIC gpu
        absl::log
        absl::log_initialize
        absl::flags_parse
        )

    target_compile_definitions(core PUBLIC ENABLE_CUDA)

else()
    message(STATUS "CUDA not found: building CPU-only fallback")
endif()
