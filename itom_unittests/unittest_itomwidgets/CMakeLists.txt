cmake_minimum_required(VERSION 3.12...3.29)
set(target_name unittest_itomwidgets)

project(${target_name})

# GUI unittests of the itomWidgets library, based on GoogleTest and the Qt Test module.
#
# The tests run without a display by default (QT_QPA_PLATFORM=offscreen is set in unittest_itomwidgets.cc,
# if the variable is not defined). To watch the tests on screen, set QT_QPA_PLATFORM to
# 'windows' (Windows) or 'xcb' (Linux) before executing unittest_itomwidgets.

option(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." ON)

set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/../..
    ${PROJECT_SOURCE_DIR}/../../gtest-1.12.1/googletest/
)

include("../../cmake/ItomBuildMacros.cmake")

# no automoc required: the tests do not declare own Q_OBJECT classes.
itom_find_package_qt(OFF Widgets Test)

if(WIN32 AND ((NOT DEFINED ${gtest_force_shared_crt}) OR (NOT ${gtest_force_shared_crt})))
    set(gtest_force_shared_crt ON CACHE BOOL "Use shared (DLL) run-time lib even when Google Test is built as static lib." FORCE)
    message(WARNING "For properly running the unittests, the CMAKE variable gtest_force_shared_crt has been forced to ON.")
endif()

# Define helper functions and macros used by Google Test.
include("../../gtest-1.12.1/googletest/cmake/internal_utils.cmake")
fix_default_compiler_settings_()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../gtest-1.12.1/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../itomWidgets
)

set(unittest_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/widgetTestHelpers.h
)

set(unittest_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/checkableComboBoxTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/collapsibleGroupBoxTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/doubleSpinBoxTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rangeSliderTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/searchBoxTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sliderWidgetTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/statusLedTest.cpp
)

# cxx_executable (from gtest) automatically adds ${target_name}.cc, which contains main().
cxx_executable(${target_name} ${CMAKE_CURRENT_SOURCE_DIR} gtest ${unittest_SOURCES} ${unittest_HEADERS})
target_link_libraries(${target_name} itomWidgets ${QT5_LIBRARIES})

set_target_properties(${target_name}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/"
)

set_target_properties(${target_name} PROPERTIES DEBUG_POSTFIX "d")

# register with ctest (only effective if enable_testing() is called in a parent directory)
add_test(NAME ${target_name} COMMAND ${target_name} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
set_tests_properties(${target_name} PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
