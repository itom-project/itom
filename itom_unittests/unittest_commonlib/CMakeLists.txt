set(target_name unittest_commonlib)
 
project(${target_name})

cmake_minimum_required(VERSION 3.0.2)

option(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." ON) 

set(CMAKE_MODULE_PATH 
    ${CMAKE_MODULE_PATH} 
    ${PROJECT_SOURCE_DIR} 
    ${PROJECT_SOURCE_DIR}/../.. 
    ${PROJECT_SOURCE_DIR}/../../gtest-1.10.0/googletest/
)

include("../../cmake/ItomBuildMacros.cmake")

find_package(OpenCV REQUIRED COMPONENTS core)

# Define helper functions and macros used by Google Test.
include("../../gtest-1.10.0/googletest/cmake/internal_utils.cmake")


#config_compiler_and_linker()  # Defined in internal_utils.cmake.
fix_default_compiler_settings_()

if(WIN32 AND NOT ${gtest_force_shared_crt})
    message("Please set gtest_force_shared_crt to ON.")
endif()


include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../gtest-1.10.0/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
)

link_directories(
    # ${CMAKE_CURRENT_BINARY_DIR}/../DataObject
    
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

set_target_properties(${target_name} PROPERTIES DEBUG_POSTFIX "d")




