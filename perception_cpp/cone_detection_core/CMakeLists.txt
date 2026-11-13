cmake_minimum_required(VERSION 3.16)
project(cone_detection_core CXX)

# C++ port of cone_detection/fused.py, exposed to Python as the
# `cone_detection_core` module (detector_backend: cpp). Builds under colcon
# (ament_cmake) or standalone: cmake -S . -B build && cmake --build build.

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Bit-for-bit with the Numba kernels: no fast-math, no fused multiply-add.
set(CDC_FP_FLAGS -O3 -ffp-contract=off -fno-fast-math -Wall -Wextra)

find_package(ament_cmake QUIET)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  execute_process(COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
                  OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  find_package(pybind11 CONFIG REQUIRED)
endif()

add_library(cone_detection_core_lib STATIC src/core.cpp)
target_include_directories(cone_detection_core_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_options(cone_detection_core_lib PRIVATE ${CDC_FP_FLAGS})

pybind11_add_module(cone_detection_core src/bindings.cpp)
target_link_libraries(cone_detection_core PRIVATE cone_detection_core_lib)
target_compile_options(cone_detection_core PRIVATE ${CDC_FP_FLAGS})

# Plain C++ unit tests (no gtest needed): ctest / colcon test.
include(CTest)
if(BUILD_TESTING)
  add_executable(test_cone_detection_core test/test_core.cpp)
  target_link_libraries(test_cone_detection_core PRIVATE cone_detection_core_lib)
  target_compile_options(test_cone_detection_core PRIVATE ${CDC_FP_FLAGS})
  add_test(NAME test_cone_detection_core COMMAND test_cone_detection_core)
endif()

if(ament_cmake_FOUND)
  find_package(ament_cmake_python REQUIRED)
  _ament_cmake_python_get_python_install_dir()
  install(TARGETS cone_detection_core DESTINATION "${PYTHON_INSTALL_DIR}")
  install(DIRECTORY include/ DESTINATION include)
  ament_package()
endif()
