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

# Tell CMake to create the helloworld executable
add_executable(codeEditor WIN32 
    main.cpp 
     
    include/codeEditor.h 
    include/manager.h
    include/panel.h
    include/mode.h
    include/textDecoration.h
    codeEditor.cpp
    mode.cpp
    managers/manager.cpp
    managers/panelsManager.h
    managers/panelsManager.cpp
    managers/textDecorationsManager.h
    managers/textDecorationsManager.cpp
    panels/panel.cpp
    decorations/textDecoration.cpp
    ${RES_SOURCES} )

# Use the Widgets module from Qt 5.
target_link_libraries(codeEditor Qt5::Widgets)