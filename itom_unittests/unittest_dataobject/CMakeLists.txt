# - itom software
# URL: http://www.uni-stuttgart.de/ito
# Copyright (C) 2020, Institut fuer Technische Optik (ITO),
# Universitaet Stuttgart, Germany
#
# This file is part of itom and its software development toolkit (SDK).
#
# itom is free software; you can redistribute it and/or modify it
# under the terms of the GNU Library General Public Licence as published by
# the Free Software Foundation; either version 2 of the Licence, or (at
# your option) any later version.
#
# itom is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
# General Public Licence for more details.
#
# You should have received a copy of the GNU Library General Public License
# along with itom. If not, see <http://www.gnu.org/licenses/>.

set(target_name unittest_dataobject)

project(${target_name})

cmake_minimum_required(VERSION 3.1...3.15)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

option(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." ON) 
set(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/../../cmake ${PROJECT_SOURCE_DIR}/../../gtest-1.6.0)

include("../../cmake/ItomBuildMacros.cmake")

add_definitions(-DITOMLIBS_SHARED -D_ITOMLIBS_SHARED)

# Define helper functions and macros used by Google Test.
include("../../gtest-1.6.0/cmake/internal_utils.cmake")

find_package(OpenCV REQUIRED COMPONENTS core)

#config_compiler_and_linker()  # Defined in internal_utils.cmake.
fix_default_compiler_settings_()

if(WIN32 AND NOT ${gtest_force_shared_crt})
    message("Please set gtest_force_shared_crt to ON.")
endif(WIN32 AND NOT ${gtest_force_shared_crt})

include_directories(
    # ${CMAKE_CURRENT_BINARY_DIR}
    # ${CMAKE_CURRENT_SOURCE_DIR}
    # ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${OpenCV_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../gtest-1.6.0/include
)

set(unittest_dataobject_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/commonChannel.h    
    ${CMAKE_CURRENT_SOURCE_DIR}/operatorComplexTest.h
    ${CMAKE_CURRENT_SOURCE_DIR}/test_global.h
)

set(unittest_dataobject_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/assignTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/addressTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/adjustROI_func_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/at_func_test1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/at_func_test2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/convertCmplxTypeToRealType_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dataObjectFreeCreate_Test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/complexDataTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dataObjectTagSpace_operator_Test
    ${CMAKE_CURRENT_SOURCE_DIR}/dataObjectTagType_Test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dataObjectTag_Test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dataTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dimsAndtype_Test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DObj_copy_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DObj_cpy_test1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/functions_ROITest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/getDataType_Test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/isZeroValue_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/locateROI_func_test.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/miscellaneous.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/numberConversion_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/operatorTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/saturate_cast_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/saturateTest_real.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/seekMat_calcNumMats_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_dataobject.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/transposeTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/makeContinuousFuncTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Iterator_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ConstIterator_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/colorTypesTest.cpp
)

cxx_executable(${target_name} ${CMAKE_CURRENT_SOURCE_DIR} gtest ${unittest_dataobject_SOURCES} ${unittest_dataobject_HEADERS})
target_link_libraries(${target_name} dataobject itomCommonLib ${OpenCV_LIBS})

set_target_properties( ${target_name}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/"
)

SET_TARGET_PROPERTIES(${target_name} PROPERTIES DEBUG_POSTFIX "d")




