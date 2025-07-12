cmake_minimum_required(VERSION 3.10)
project(NerualNetwork)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS}")


add_executable(main
    main.cc
    mnist.cc
)
target_compile_options(main PRIVATE -Wall -Werror -Wpedantic)

add_executable(benchmark
    benchmark.cc
    mnist.cc
)
