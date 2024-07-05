cmake_minimum_required(VERSION 3.16)
project(ttnn-matmul-corruption-demo CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if("$ENV{TT_METAL_HOME}" STREQUAL "")
    message(FATAL_ERROR "TT_METAL_HOME is not set")
endif()
if("$ENV{ARCH_NAME}" STREQUAL "")
    message(FATAL_ERROR "ARCH_NAME is not set")
endif()

message($ENV{TT_METAL_HOME}/tt_metal/third_party/src/firmware/riscv/$ENV{ARCH_NAME})
add_executable(ttnn-matmul-corruption-demo ttnn-matmul-corruption-demo.cpp)
target_include_directories(ttnn-matmul-corruption-demo PRIVATE
    # Metalium
    $ENV{TT_METAL_HOME}
    $ENV{TT_METAL_HOME}/tt_metal
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd
    $ENV{TT_METAL_HOME}/tt_metal/third_party/fmt
    $ENV{TT_METAL_HOME}/tt_metal/hw/inc/$ENV{ARCH_NAME}
    $ENV{TT_METAL_HOME}/tt_metal/hw/inc/
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/src/firmware/riscv/$ENV{ARCH_NAME}

    # TTNN
    $ENV{TT_METAL_HOME}/ttnn/cpp
    $ENV{TT_METAL_HOME}/tt_eager
    $ENV{TT_METAL_HOME}/tt_metal/third_party/magic_enum

)
target_link_directories(ttnn-matmul-corruption-demo PRIVATE
    $ENV{TT_METAL_HOME}/build/lib
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")

# TODO: In the future TTNN can optionally not depend on Python
find_package(Python REQUIRED Development)
target_link_libraries(ttnn-matmul-corruption-demo PRIVATE
    # Metalium
    tt_metal
    yaml-cpp
    c++abi
    c++

    # TTNN
    tt_eager
    Python::Python
    $ENV{TT_METAL_HOME}/build/lib/_ttnn.so
)

target_compile_definitions(ttnn-matmul-corruption-demo PRIVATE
    FMT_HEADER_ONLY
)

target_compile_options(ttnn-matmul-corruption-demo PRIVATE -stdlib=libc++)

target_precompile_headers(ttnn-matmul-corruption-demo PRIVATE pch.hpp)