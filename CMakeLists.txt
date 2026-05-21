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

# The tensor module is mostly a header-only template library; the only
# compiled unit is the CUDA dispatch layer in cuda_ops.cu. Anything that
# links cue_tensor picks up the include directory and libcudart.
add_library(cue_tensor STATIC tensor/cuda_ops.cu)
target_include_directories(cue_tensor PUBLIC tensor)
set_target_properties(cue_tensor PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

enable_testing()
add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE cue_tensor)
target_compile_options(test_tensor PRIVATE -Wall -Wextra)
add_test(NAME tensor COMMAND test_tensor)
