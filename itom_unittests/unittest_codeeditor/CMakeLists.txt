cmake_minimum_required(VERSION 3.12...3.29)
set(target_name unittest_codeeditor)

project(${target_name})

option(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." ON)

set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/../..
    ${PROJECT_SOURCE_DIR}/../../gtest-1.12.1/googletest/
)

include("../../cmake/ItomBuildMacros.cmake")

# the code editor is part of the qitom executable (no library). Therefore the core
# of the editor (editor widget, managers, panels, syntax highlighters, fold detectors)
# is compiled directly into this test. Python dependent modes are not part of the test.
itom_find_package_qt(ON Core Gui Widgets)

if(WIN32 AND ((NOT DEFINED ${gtest_force_shared_crt}) OR (NOT ${gtest_force_shared_crt})))
    set(gtest_force_shared_crt ON CACHE BOOL "Use shared (DLL) run-time lib even when Google Test is built as static lib." FORCE)
    message(WARNING "For properly running the unittests, the CMAKE variable gtest_force_shared_crt has been forced to ON.")
endif()

# Define helper functions and macros used by Google Test.
include("../../gtest-1.12.1/googletest/cmake/internal_utils.cmake")

#config_compiler_and_linker()  # Defined in internal_utils.cmake.
fix_default_compiler_settings_()

set(QITOM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../Qitom)
set(CODEEDITOR_DIR ${QITOM_DIR}/codeEditor)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../gtest-1.12.1/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
    ${QITOM_DIR}
    ${CODEEDITOR_DIR}
)

set(codeeditor_HEADERS
    ${QITOM_DIR}/helper/charsetEncoding.h
    ${QITOM_DIR}/helper/editorFileType.h
    ${CODEEDITOR_DIR}/codeCheckerItem.h
    ${CODEEDITOR_DIR}/codeEditor.h
    ${CODEEDITOR_DIR}/delayJobRunner.h
    ${CODEEDITOR_DIR}/mode.h
    ${CODEEDITOR_DIR}/panel.h
    ${CODEEDITOR_DIR}/textBlockUserData.h
    ${CODEEDITOR_DIR}/textDecoration.h
    ${CODEEDITOR_DIR}/toolTip.h
    ${CODEEDITOR_DIR}/foldDetector/charBasedFoldDetector.h
    ${CODEEDITOR_DIR}/foldDetector/foldDetector.h
    ${CODEEDITOR_DIR}/foldDetector/indentFoldDetector.h
    ${CODEEDITOR_DIR}/managers/manager.h
    ${CODEEDITOR_DIR}/managers/modesManager.h
    ${CODEEDITOR_DIR}/managers/panelsManager.h
    ${CODEEDITOR_DIR}/managers/textDecorationsManager.h
    ${CODEEDITOR_DIR}/modes/caretLineHighlight.h
    ${CODEEDITOR_DIR}/modes/lineBackgroundMarker.h
    ${CODEEDITOR_DIR}/modes/occurrences.h
    ${CODEEDITOR_DIR}/modes/symbolMatcherMode.h
    ${CODEEDITOR_DIR}/panels/foldingPanel.h
    ${CODEEDITOR_DIR}/syntaxHighlighter/codeEditorStyle.h
    ${CODEEDITOR_DIR}/syntaxHighlighter/genericSyntaxHighlighter.h
    ${CODEEDITOR_DIR}/syntaxHighlighter/syntaxHighlighterBase.h
    ${CODEEDITOR_DIR}/utils/utils.h
)

set(codeeditor_SOURCES
    ${QITOM_DIR}/helper/charsetEncoding.cpp
    ${QITOM_DIR}/helper/editorFileType.cpp
    ${CODEEDITOR_DIR}/codeCheckerItem.cpp
    ${CODEEDITOR_DIR}/codeEditor.cpp
    ${CODEEDITOR_DIR}/delayJobRunner.cpp
    ${CODEEDITOR_DIR}/mode.cpp
    ${CODEEDITOR_DIR}/panel.cpp
    ${CODEEDITOR_DIR}/textBlockUserData.cpp
    ${CODEEDITOR_DIR}/textDecoration.cpp
    ${CODEEDITOR_DIR}/toolTip.cpp
    ${CODEEDITOR_DIR}/foldDetector/charBasedFoldDetector.cpp
    ${CODEEDITOR_DIR}/foldDetector/foldDetector.cpp
    ${CODEEDITOR_DIR}/foldDetector/indentFoldDetector.cpp
    ${CODEEDITOR_DIR}/managers/manager.cpp
    ${CODEEDITOR_DIR}/managers/modesManager.cpp
    ${CODEEDITOR_DIR}/managers/panelsManager.cpp
    ${CODEEDITOR_DIR}/managers/textDecorationsManager.cpp
    ${CODEEDITOR_DIR}/modes/caretLineHighlight.cpp
    ${CODEEDITOR_DIR}/modes/lineBackgroundMarker.cpp
    ${CODEEDITOR_DIR}/modes/occurrences.cpp
    ${CODEEDITOR_DIR}/modes/symbolMatcherMode.cpp
    ${CODEEDITOR_DIR}/panels/foldingPanel.cpp
    ${CODEEDITOR_DIR}/syntaxHighlighter/codeEditorStyle.cpp
    ${CODEEDITOR_DIR}/syntaxHighlighter/genericSyntaxHighlighter.cpp
    ${CODEEDITOR_DIR}/syntaxHighlighter/syntaxHighlighterBase.cpp
    ${CODEEDITOR_DIR}/utils/utils.cpp
)

set(unittest_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/unittest_codeeditor.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditorTestHelper.h
)

set(unittest_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/appManagementStub.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codeEditorTestHelper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/longLinePaintTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/symbolMatcherTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/occurrencesTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/charsetEncodingTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/editorFileTypeTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/genericSyntaxHighlighterTest.cpp
)

cxx_executable(${target_name} ${CMAKE_CURRENT_SOURCE_DIR} gtest
    ${unittest_SOURCES} ${unittest_HEADERS} ${codeeditor_SOURCES} ${codeeditor_HEADERS})
target_link_libraries(${target_name} ${QT5_LIBRARIES})

set_target_properties( ${target_name}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/"
)

set_target_properties(${target_name} PROPERTIES DEBUG_POSTFIX "d")
