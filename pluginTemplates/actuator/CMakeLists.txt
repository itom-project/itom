###################################################################
###################################################################
#    CMake Template for a plugin of itom
#    
#    You can use this template, use it in your plugins, modify it,
#    copy it and distribute it without any license restrictions.
###################################################################
###################################################################


# this should be the first line of your project.
cmake_minimum_required(VERSION 3.1...3.15)

# This looks for a valid itom SDK...
find_path(ITOM_SDK_DIR "ItomBuildMacros"
    HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../../build/itom/SDK")
if(NOT ITOM_SDK_DIR)
    message(ERROR "ITOMSDK required but not found, set ITOM_SDK_DIR manually")
    break()
endif()
# ...which contains useful scripts for completing this file
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${ITOM_SDK_DIR}/cmake)

# ... for example this one
include(ItomBuildMacros)

#here you give your project a unique name
set(target_name MyActuator_Name)

# and this does some magic for you, as setting some common vars and 
# passing some switches to your project you can use in sourcecode
itom_init_plugin_library(${target_name})

#################################################################
# Input elements for CMake GUI (Checkboxes, Pathes, Strings...)
#################################################################
# BUILD_TARGET64 is set according to bittage of selected compiler
# You may select loaded libraries(32 vs 64 bit) according to the Value of this
# option(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." ON)


#################################################################
# Automatic package detection
#   add here find_package commands for searching for 3rd party
#   libraries
#
#   for detecting Qt, use itom_find_package_qt instead of the
#   native command.
#################################################################

# the itom SDK needs to be detected, use the COMPONENTS keyword
# to define which library components are needed. Possible values
# are:
# - dataobject for the ito::DataObject (usually required)
# - itomCommonLib (RetVal, Param,...) (required)
# - itomCommonQtLib (AddInInterface,...) (required)
# - itomWidgets (further widgets) (may be used in dock widget...)
# - pointcloud (pointCloud, point, polygonMesh...) (optional)
# if no components are indicated, all components above are used
find_package(ITOM_SDK COMPONENTS dataobject itomCommonLib itomCommonQtLib REQUIRED)
#find_package(OpenCV COMPONENTS core REQUIRED) #if you require openCV
# indicate all components that are required (e.g. core, imgproc...), 
#       if the dataobject is included in the ITOM_SDK components,
# the OpenCV core component is detected there and the necessary include
#       directories and libraries to link agains are contained in
# ITOM_SDK_LIBRARIES and ITOM_SDK_INCLUDE_DIRS

#usage of FIND_PACKAGE_QT(automoc component1, component2, ...)
# automoc is ON or OFF and only relevant for Qt5, usually set it to ON
# possible components are: OpenGL,Core,Designer,Xml,Svg,Sql,Network,UiTools,Widgets,PrintSupport,LinguistTools...
itom_find_package_qt(ON Core Widgets LinguistTools)

#################################################################
# General settings and preprocessor settings
#################################################################





#################################################################
# List of include directories
#
# Hint: necessary Qt include directories are automatically added
#  via the FIND_PACKAGE macro above
#################################################################
include_directories(
    ${CMAKE_CURRENT_BINARY_DIR} #build directory of this plugin (recommended)
    ${CMAKE_CURRENT_SOURCE_DIR} #source directory of this plugin (recommended)
    #add further include directories here
)

#################################################################
# List of linker directories
#
# Hint: libraries detected using FIND_PACKAGE usually provide
#  all necessary libraries in a specific variable (e.g.
#  ${OpenCV_LIBS} or ${ITOM_SDK_LIBRARIES}). These variables
#  already contain absolute pathes, therefore no link directory
#  needs to be set for them. Simply add these variables to
#  the link target command below.
#################################################################
link_directories(
    #add all linker directories
)

#################################################################
# List of header files, source files, ui files and rcc files
#
# Add all header files to the plugin_HEADERS list.
# Add all source (cpp,...) files to the plugin_SOURCES list.
# Add all ui-files (Qt-Designer layouts) to the plugin_UI list.
#
# Use absolute pathes, e.g. using one of the following variables:
#
# ${ITOM_SDK_INCLUDE_DIR} is the include directory of itom SDK
# ${CMAKE_CURRENT_SOURCE_DIR} is the source directory of this plugin
# ${CMAKE_CURRENT_BINARY_DIR} is the build directory of this plugin
#
#################################################################
set(plugin_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/myActuator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/dialogMyActuator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/dockWidgetMyActuator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pluginVersion.h
    ${CMAKE_CURRENT_BINARY_DIR}/gitVersion.h #provided by the init script, cotains currently checked out git tag
    #add further header files (absolute pathes e.g. using CMAKE_CURRENT_SOURCE_DIR)
)

set(plugin_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/dialogMyActuator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dockWidgetMyActuator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/myActuator.cpp
    #add further source files here
)

#Append rc file to the source files for adding information about the plugin
# to the properties of the DLL under Visual Studio.
if(MSVC)
    list(APPEND plugin_SOURCES ${ITOM_SDK_INCLUDE_DIR}/../pluginLibraryVersion.rc)
endif()

set(plugin_UI
    ${CMAKE_CURRENT_SOURCE_DIR}/dialogMyActuator.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/dockWidgetMyActuator.ui
)

set(plugin_RCC
    #add absolute pathes to any *.qrc resource files here
)

#################################################################
# Qt related pre-processing of the files above
# (These methods create the moc, rcc and uic process.)
#################################################################
qt5_wrap_ui(plugin_UI_MOC ${plugin_UI})
qt5_add_resources(plugin_RCC_MOC ${plugin_RCC})


#################################################################
# Group files in their original folder structure (MSVC only)
# If you have some header and source files in a specific
# subfolder, you can even have this subfolder in your
# IDE (mainly Visual Studio supports this). Then call 
# itom_add_source_group(directoryName) for each subdirectory.
#
# HINT: This command does nothing for IDE different than MSVC.
#################################################################
#itom_add_source_group(subdirectory)


#################################################################
# Compile and link the plugin library
# 
#################################################################

#search for all existing translation files in the translation subfolder
file(GLOB EXISTING_TRANSLATION_FILES "translation/*.ts") 

#add all (generated) header and source files to the library (these files are compiled then)
add_library(${target_name} SHARED
    ${plugin_SOURCES}
    ${plugin_HEADERS}
    ${plugin_HEADERS_MOC}
    ${plugin_UI_MOC}
    ${plugin_RCC_MOC}
    ${EXISTING_TRANSLATION_FILES}
    )

#link the compiled library
#append all libraries this plugin should be linked to at the end of the target_link_libraries command
# Important variables are:
# ${ITOM_SDK_LIBRARIES} -> all necessary libraries from find_package(ITOM_SDK)
# -> all necessary libraries from FIND_PACKAGE_QT (Qt4 or Qt5)
# ${OpenCV_LIBS} -> all necessary libraries opencv libraries from find_package(OpenCV)
#
# if you want to link against one library whose directory is already added to link_directories above
# simply add its filename without suffix (*.lib, *.so...). This is automatically done by CMake

target_link_libraries(${target_name}
    ${ITOM_SDK_LIBRARIES}
    ${QT5_LIBRARIES}
    ${VISUALLEAKDETECTOR_LIBRARIES}
    )

#################################################################
# Plugin Translation
#################################################################
set(FILES_TO_TRANSLATE ${plugin_SOURCES} ${plugin_HEADERS} ${plugin_UI})
itom_library_translation(QM_FILES ${target_name}
            ${UPDATE_TRANSLATIONS}
            "${EXISTING_TRANSLATION_FILES}"
            ITOM_LANGUAGES
            "${FILES_TO_TRANSLATE}")

#################################################################
# Plugin Documentation
# 
# In the source directory of the plugin can be a subfolder 'docs'.
# This folder can contain one or more *.rst files with the docu-
# mentation of the plugin. CMake organizes the rest if you 
# indicate the name of the main documentation file(without
# suffix rst) in the following command:
# 
# PLUGIN_DOCUMENTATION(${target_name} nameOfTheFile)
#################################################################
itom_configure_plugin_documentation(${target_name} myGrabber)

#################################################################
# Post-Build Copy Operations
# 
# itom is able to force a post-build process that copies
# different files, like the currently created library, to
# other destination pathes. This is done in this section.
# At first pairs of sources and destinations are added
# to the lists COPY_SOURCES and COPY_DESTINATIONS.
# Afterwards, the post-build process is generated using
# itom_post_build_copy_files.
#
# The following macros can be used to fill up the source
# and destination list:
#
# ADD_PLUGINLIBRARY_TO_COPY_LIST 
# - this is necessary for each plugin such that the library
#   is automatically copied to the plugins folder of
#   the itom build directory.
#
# ADD_QM_FILES_TO_COPY_LIST
# - installs the generated translation files (qm) at the
#   right place in the itom build directory as well.
#
#################################################################
set(COPY_SOURCES "")
set(COPY_DESTINATIONS "")

itom_add_pluginlibrary_to_copy_list(${target_name} COPY_SOURCES COPY_DESTINATIONS)
itom_add_plugin_qm_files_to_copy_list(${target_name} QM_FILES COPY_SOURCES COPY_DESTINATIONS)

itom_post_build_copy_files(${target_name} COPY_SOURCES COPY_DESTINATIONS)

#if you want to copy one or more files to the lib-folder of
# the itom build directory, use the following macro:
#
# POST_BUILD_COPY_FILE_TO_LIB_FOLDER(${target}, ${listOfFiles})

