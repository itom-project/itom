# Note: CMake support is community-based. The maintainers do not use CMake
# internally.

cmake_minimum_required(VERSION 3.12...3.24)

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

project(googletest-distribution)
set(GOOGLETEST_VERSION 1.10.0)

if (CMAKE_VERSION VERSION_LESS "3.1")
  add_definitions(-std=c++11)
else()
  if(BUILD_QTVERSION STREQUAL "Qt6")
    set(CMAKE_CXX_STANDARD 17)
  elseif(BUILD_QTVERSION STREQUAL "Qt5")
    set(CMAKE_CXX_STANDARD 11)
  endif(BUILD_QTVERSION STREQUAL "Qt6")
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  if(NOT CYGWIN)
    set(CMAKE_CXX_EXTENSIONS OFF)
  endif()
endif()

enable_testing()

include(CMakeDependentOption)
include(GNUInstallDirs)

add_subdirectory( googletest )
