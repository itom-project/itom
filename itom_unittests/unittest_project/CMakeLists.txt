cmake_minimum_required(VERSION 3.12...3.29)
set(target_name unittest_project)

project(${target_name})

option(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." ON)

set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/../..
    ${PROJECT_SOURCE_DIR}/../../gtest-1.12.1/googletest/
)

include("../../cmake/ItomBuildMacros.cmake")

# Qitom/organizer/projectFile and Qitom/models/projectTreeModel are part of the qitom
# executable (no library). Therefore they are compiled directly into this test
# (the model needs editorFileType for the icons). QtTest provides QAbstractItemModelTester.
itom_find_package_qt(ON Core Gui Widgets Test)

find_package(OpenCV REQUIRED COMPONENTS core)

if(WIN32 AND ((NOT DEFINED ${gtest_force_shared_crt}) OR (NOT ${gtest_force_shared_crt})))
    set(gtest_force_shared_crt ON CACHE BOOL "Use shared (DLL) run-time lib even when Google Test is built as static lib." FORCE)
    message(WARNING "For properly running the unittests, the CMAKE variable gtest_force_shared_crt has been forced to ON.")
endif()

# Define helper functions and macros used by Google Test.
include("../../gtest-1.12.1/googletest/cmake/internal_utils.cmake")

#config_compiler_and_linker()  # Defined in internal_utils.cmake.
fix_default_compiler_settings_()

set(QITOM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../Qitom)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../gtest-1.12.1/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
    ${QITOM_DIR}
)

set(qitom_HEADERS
    ${QITOM_DIR}/helper/editorFileType.h
    ${QITOM_DIR}/models/projectTreeModel.h
    ${QITOM_DIR}/organizer/projectFile.h
)

set(qitom_SOURCES
    ${QITOM_DIR}/helper/editorFileType.cpp
    ${QITOM_DIR}/models/projectTreeModel.cpp
    ${QITOM_DIR}/organizer/projectFile.cpp
)

set(unittest_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/unittest_project.cc
)

set(unittest_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/appManagementStub.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/projectFileTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/projectTreeModelTest.cpp
)

cxx_executable(${target_name} ${CMAKE_CURRENT_SOURCE_DIR} gtest
    ${unittest_SOURCES} ${unittest_HEADERS} ${qitom_SOURCES} ${qitom_HEADERS})
target_link_libraries(${target_name} itomCommonLib ${QT5_LIBRARIES} ${OpenCV_LIBS})

set_target_properties( ${target_name}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/"
)

set_target_properties(${target_name} PROPERTIES DEBUG_POSTFIX "d")
