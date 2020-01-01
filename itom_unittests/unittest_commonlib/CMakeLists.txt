SET (target_name unittest_commonlib)
 
project(${target_name})

cmake_minimum_required(VERSION 2.8)

OPTION(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." ON) 
SET(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.")

SET (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/../.. ${PROJECT_SOURCE_DIR}/../../gtest-1.6.0)

include("../../ItomBuildMacros.cmake")

find_package(OpenCV REQUIRED COMPONENTS core)

# Define helper functions and macros used by Google Test.
include("../../gtest-1.6.0/cmake/internal_utils.cmake")


#config_compiler_and_linker()  # Defined in internal_utils.cmake.
fix_default_compiler_settings_()

IF(WIN32 AND NOT ${gtest_force_shared_crt})
    message("Please set gtest_force_shared_crt to ON.")
ENDIF(WIN32 AND NOT ${gtest_force_shared_crt})


INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../gtest-1.6.0/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
    ${OpenCV_DIR}/include
)

LINK_DIRECTORIES(
    # ${CMAKE_CURRENT_BINARY_DIR}/../DataObject
    ${OpenCV_DIR}/lib
)

set(unittest_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/unittest_commonlib.cc
)

set(unittest_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/byteArrayTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/retValTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/paramTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/colorTest.cpp
)

cxx_executable(${target_name} ${CMAKE_CURRENT_SOURCE_DIR} gtest ${unittest_SOURCES} ${unittest_HEADERS})
target_link_libraries(${target_name} itomCommonLib ${OpenCV_LIBS})

set_target_properties( ${target_name}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/"
)

SET_TARGET_PROPERTIES(${target_name} PROPERTIES DEBUG_POSTFIX "d")




