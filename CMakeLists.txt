cmake_minimum_required(VERSION 3.18)
project(cue LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Single archive holds the CUDA dispatch layer plus the pure-C++ nn and
# data modules. Bundling them avoids the extra device-link step that
# downstream-only CXX libraries trigger when they transitively depend on
# CUDA-separable code.
add_library(cue STATIC
    tensor/cuda_ops.cu
    nn/nn.cpp
    data/dataset.cpp
)
target_include_directories(cue PUBLIC tensor nn data)
set_target_properties(cue PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(cue PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

enable_testing()

foreach(t IN ITEMS test_tensor test_nn test_data test_train_iris)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE cue)
    target_compile_options(${t} PRIVATE -Wall -Wextra)
    add_test(NAME ${t} COMMAND ${t})
endforeach()

target_compile_definitions(test_train_iris
    PRIVATE CUE_DATA_DIR="${CMAKE_SOURCE_DIR}/data")
