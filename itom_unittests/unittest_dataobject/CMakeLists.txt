SET (target_name unittest_dataobject)

project(${target_name})

cmake_minimum_required(VERSION 2.8)

OPTION(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." OFF) 
SET(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.")
OPTION(BUILD_ITOMLIBS_SHARED "Build unittest for itom (including gtest)." OFF)

SET (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/../.. ${PROJECT_SOURCE_DIR}/../../gtest-1.6.0)

include("../../ItomBuildMacros.cmake")

IF (BUILD_ITOMLIBS_SHARED)
    ADD_DEFINITIONS(-DITOMLIBS_SHARED -D_ITOMLIBS_SHARED)
ENDIF (BUILD_ITOMLIBS_SHARED)

# Define helper functions and macros used by Google Test.
include("../../gtest-1.6.0/cmake/internal_utils.cmake")

find_package(OpenCV REQUIRED COMPONENTS core)

#config_compiler_and_linker()  # Defined in internal_utils.cmake.
fix_default_compiler_settings_()

IF(WIN32 AND NOT ${gtest_force_shared_crt})
    message("Please set gtest_force_shared_crt to ON.")
ENDIF(WIN32 AND NOT ${gtest_force_shared_crt})

# IF (BUILD_UNICODE)
    # ADD_DEFINITIONS(-DUNICODE -D_UNICODE)
# ENDIF (BUILD_UNICODE)

# ADD_DEFINITIONS(-DCMAKE)

# # default build types are None, Debug, Release, RelWithDebInfo and MinRelSize
# IF (DEFINED CMAKE_BUILD_TYPE)
    # SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
# ELSE(CMAKE_BUILD_TYPE)
    # SET (CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
# ENDIF (DEFINED CMAKE_BUILD_TYPE)

INCLUDE_DIRECTORIES(
    # ${CMAKE_CURRENT_BINARY_DIR}
    # ${CMAKE_CURRENT_SOURCE_DIR}
    # ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${OpenCV_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../gtest-1.6.0/include
)

# LINK_DIRECTORIES(
    # ${CMAKE_CURRENT_BINARY_DIR}/../DataObject
    # ${OpenCV_DIR}/lib
# )

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
    ${CMAKE_CURRENT_SOURCE_DIR}/makeContinuousFuncTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Iterator_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ConstIterator_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/colorTypesTest.cpp
)

# #add filters to visual studio
# ADD_SOURCE_GROUP(api)
# ADD_SOURCE_GROUP(helper)
# ADD_SOURCE_GROUP(models)
# ADD_SOURCE_GROUP(organizer)
# ADD_SOURCE_GROUP(python)
# ADD_SOURCE_GROUP(ui)
# ADD_SOURCE_GROUP(widgets)
# #end add filters to visual studio

cxx_executable(${target_name} ${CMAKE_CURRENT_SOURCE_DIR} gtest ${unittest_dataobject_SOURCES} ${unittest_dataobject_HEADERS})
target_link_libraries(${target_name} dataobject itomCommonLib ${OpenCV_LIBS})

set_target_properties( ${target_name}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/"
)

SET_TARGET_PROPERTIES(${target_name} PROPERTIES DEBUG_POSTFIX "d")




