cmake_minimum_required(VERSION 3.16)
project(nanotorch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

#pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v3.0.1
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(nanotorch src/nanotorch.cpp)

if (MSVC)
  target_compile_options(nanotorch PRIVATE /W4)
else()
  target_compile_options(nanotorch PRIVATE -Wall -Wextra -Wpedantic)
endif()
