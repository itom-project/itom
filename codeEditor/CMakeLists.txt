cmake_minimum_required(VERSION 2.8.11)
  
project(codeEditor)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# Find the QtWidgets library
find_package(Qt5Widgets)

include_directories(
    include
)

#QT5_ADD_RESOURCES( RES_SOURCES customcompleter.qrc)
SET (HEADERS
    utils/utils.h
    include/codeEditor.h 
    include/manager.h
    include/panel.h
    include/mode.h
    include/textDecoration.h
    include/foldDetector.h
    include/delayJobRunner.h
    syntaxHighlighter/syntaxHighlighterBase.h
    syntaxHighlighter/pythonSyntaxHighlighter.h
    managers/panelsManager.h
    managers/textDecorationsManager.h
    managers/modesManager.h
    modes/caretLineHighlight.h
    modes/symbolMatcherMode.h
)

SET (SOURCES
    main.cpp 
    codeEditor.cpp
    mode.cpp
    foldDetector.cpp
    syntaxHighlighter/syntaxHighlighterBase.cpp
    syntaxHighlighter/pythonSyntaxHighlighter.cpp
    delayJobRunner.cpp
    managers/manager.cpp
    managers/panelsManager.cpp
    managers/textDecorationsManager.cpp
    managers/modesManager.cpp
    panels/panel.cpp
    modes/caretLineHighlight.cpp
    modes/symbolMatcherMode.cpp
    decorations/textDecoration.cpp
    utils/utils.cpp
)

# Tell CMake to create the helloworld executable
add_executable(codeEditor WIN32 
    ${HEADERS}
    ${SOURCES}
    ${RES_SOURCES} )

# Use the Widgets module from Qt 5.
target_link_libraries(codeEditor Qt5::Widgets)