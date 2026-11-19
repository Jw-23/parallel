cmake_minimum_required(VERSION 3.24)
project(parallel VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Boost REQUIRED CONFIG)
find_package(Threads REQUIRED)
add_library(parallel src/parrallel.cpp)
target_compile_features(parallel PUBLIC cxx_std_14)
target_include_directories(parallel PUBLIC include)
target_link_libraries(parallel PUBLIC Boost::headers Threads::Threads)

set(parallel_tidy_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/parrallel.cpp")
include(CTest)
include(FetchContent)
if(BUILD_TESTING)
  # GoogleTest 1.16 supports C++14.
  set(BUILD_GMOCK OFF)
  set(INSTALL_GTEST OFF)
  FetchContent_Declare(googletest
    URL https://codeload.github.com/google/googletest/tar.gz/refs/tags/v1.16.0
    URL_HASH SHA256=78c676fc63881529bf97bf9d45948d905a66833fbfa5318ea2cd7478cb98f399)
  FetchContent_MakeAvailable(googletest)
  add_executable(parallel_tests tests/parallel_tests.cpp)
  target_link_libraries(parallel_tests PRIVATE parallel GTest::gtest_main)
  # GoogleTest 1.16's TEST macro uses [[maybe_unused]] with Clang in C++14 mode.
  target_compile_options(parallel_tests PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-Wno-c++17-attribute-extensions>)
  include(GoogleTest)
  gtest_discover_tests(parallel_tests)
  list(APPEND parallel_tidy_sources "${CMAKE_CURRENT_SOURCE_DIR}/tests/parallel_tests.cpp")
endif()

option(PARALLEL_BUILD_BENCHMARKS "Build the scheduler benchmarks" OFF)
if(PARALLEL_BUILD_BENCHMARKS)
  set(BENCHMARK_ENABLE_TESTING OFF)
  set(BENCHMARK_ENABLE_INSTALL OFF)
  FetchContent_Declare(benchmark
    URL https://codeload.github.com/google/benchmark/tar.gz/refs/tags/v1.9.0
    URL_HASH SHA256=35a77f46cc782b16fac8d3b107fbfbb37dcd645f7c28eee19f3b8e0758b48994)
  FetchContent_MakeAvailable(benchmark)
  add_executable(parallel_benchmarks benchmarks/scheduler_benchmark.cpp)
  target_link_libraries(parallel_benchmarks PRIVATE parallel benchmark::benchmark_main)
  list(APPEND parallel_tidy_sources "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/scheduler_benchmark.cpp")
endif()

find_program(CLANG_TIDY_EXECUTABLE NAMES clang-tidy
  PATHS /opt/homebrew/opt/llvm/bin /usr/local/opt/llvm/bin)
foreach(target IN ITEMS clang-tidy clang-tidy-fix)
  if(CLANG_TIDY_EXECUTABLE)
    set(tidy_options)
    if(target STREQUAL "clang-tidy-fix")
      list(APPEND tidy_options --fix)
    endif()
    add_custom_target(${target}
      COMMAND "${CLANG_TIDY_EXECUTABLE}" "-p=${CMAKE_BINARY_DIR}"
              ${tidy_options} ${parallel_tidy_sources}
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      VERBATIM)
  else()
    add_custom_target(${target}
      COMMAND "${CMAKE_COMMAND}" -E echo "Install clang-tidy or set CLANG_TIDY_EXECUTABLE and reconfigure."
      COMMAND "${CMAKE_COMMAND}" -E false
      VERBATIM)
  endif()
endforeach()
