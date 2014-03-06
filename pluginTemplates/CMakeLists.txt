#Replace PluginName by the unique name of your plugin
SET (target_name PluginName)

project(${target_name})

cmake_minimum_required(VERSION 2.8)

OPTION(BUILD_UNICODE "Build with unicode charset if set to ON, else multibyte charset." ON)
OPTION(BUILD_SHARED_LIBS "Build shared library." ON)
OPTION(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." OFF)
OPTION(UPDATE_TRANSLATIONS "Update source translation translation/*.ts files (WARNING: make clean will delete the source .ts files! Danger!)")
SET (ITOM_SDK_DIR "" CACHE PATH "base path to itom_sdk")
SET (CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.")
SET (ITOM_LANGUAGES "de" CACHE STRING "semicolon separated list of languages that should be created (en must not be given since it is the default)")

SET (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${ITOM_SDK_DIR})

IF(BUILD_SHARED_LIBS)
	SET(LIBRARY_TYPE SHARED)
ELSE(BUILD_SHARED_LIBS)
	SET(LIBRARY_TYPE STATIC)
ENDIF(BUILD_SHARED_LIBS)

# Usually, your plugin depends on the SDK of itom, Qt4 and OpenCV.
# Therefore a find_package command for these libraries is already inserted.
# You can always add further library dependencies in this section.
find_package(ITOM_SDK REQUIRED)
find_package(Qt4 REQUIRED)
find_package(OpenCV COMPONENTS core REQUIRED)
#add here further packages, if you need some

#This commands includes the important macro of itom's SDK.
include("${ITOM_SDK_DIR}/ItomBuildMacros.cmake")

ADD_DEFINITIONS(${QT_DEFINITIONS})
IF (BUILD_UNICODE)
    ADD_DEFINITIONS(-DUNICODE -D_UNICODE)
ENDIF (BUILD_UNICODE)
ADD_DEFINITIONS(-DCMAKE)

# Qt consists of a set of components. These components are provided by findQt4, that
# is included by the find_package-command above. You can set further components to true,
# if you need them.
SET (QT_USE_QTXML TRUE)

# default build types are None, Debug, Release, RelWithDebInfo and MinRelSize
IF (DEFINED CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ELSE(CMAKE_BUILD_TYPE)
	SET (CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ENDIF (DEFINED CMAKE_BUILD_TYPE)

# here you can add some include-directories for your plugin.
# #include commands in your source code can then be relative with
# respect to entries in the following list. External libraries, found by find_package
# define their proper variables, that can also be used as include-directories.
INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${QT_QTCORE_INCLUDE_DIR}
    ${ITOM_SDK_INCLUDE_DIR}
    ${OpenCV_DIR}/include
 )

 # in the same way, you can add some linker search pathes in the following list.
LINK_DIRECTORIES(
    ${OpenCV_DIR}/lib
)

INCLUDE(${QT_USE_FILE})

# in this important section, define all header-files of your plugin. Use
# ${CMAKE_CURRENT_SOURCE_DIR} in order to get the path of the source directory of your
# plugin, or ${ITOM_SDK_INCLUDE_DIR} in order to access the files in itom's SDK
set(plugin_HEADERS
    ${ITOM_SDK_INCLUDE_DIR}/common/addInGrabber.h
    ${ITOM_SDK_INCLUDE_DIR}/common/addInInterface.h
    ${ITOM_SDK_INCLUDE_DIR}/common/helperCommon.h
    ${ITOM_SDK_INCLUDE_DIR}/common/sharedStructures.h
    ${ITOM_SDK_INCLUDE_DIR}/common/sharedStructuresQt.h
    #${CMAKE_CURRENT_SOURCE_DIR}/dialogPluginName.h
    #${CMAKE_CURRENT_SOURCE_DIR}/dockWidgetPluginName.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pluginName.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pluginVersion.h
)

QT4_WRAP_CPP_ITOM(plugin_HEADERS_MOC ${plugin_HEADERS})

# if your plugin contains some ui-files (Qt interface descriptions), put their
# path in the following list
set(plugin_ui
    #${CMAKE_CURRENT_SOURCE_DIR}/dialogCameraPlugin.ui
    #${CMAKE_CURRENT_SOURCE_DIR}/dockWidgetCameraPlugin.ui
)

QT4_WRAP_UI_ITOM(plugin_ui_MOC ${plugin_ui})

# in the same way than for the list plugin_HEADERS, fill in the following list
# the source files of your plugin
set(plugin_SOURCES 
    #${CMAKE_CURRENT_SOURCE_DIR}/dialogPluginName.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/dockWidgetPluginName.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pluginName.cpp
)

#If you want to use automatical metadata for dlls under windows use the following if-case.
if(MSVC)
    list(APPEND plugin_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/version.rc)
endif(MSVC)

#handle translations BEGIN STEP 1
set (FILES_TO_TRANSLATE ${plugin_SOURCES} ${plugin_HEADERS} ${plugin_ui})
set (TRANSLATION_OUTPUT_FILES)
set (TRANSLATIONS_FILES)

file (GLOB EXISTING_TRANSLATION_FILES "translation/*.ts")
#handle translations END STEP 1

ADD_LIBRARY(${target_name} ${LIBRARY_TYPE} ${plugin_SOURCES} ${plugin_HEADERS} ${plugin_HEADERS_MOC} ${plugin_ui_MOC} ${plugin_rcc_MOC})

# here, all components of the plugin are linked. Add in this list external libraries
# that your plugin must link against.
TARGET_LINK_LIBRARIES(${target_name} ${QT_LIBRARIES} ${QT_QTUITOOLS_LIBRARY} ${QT_QTXML_LIBRARY} ${OpenCV_LIBS} ${ITOM_SDK_dataobject_LIBRARY} ${VISUALLEAKDETECTOR_LIBRARIES} ${ITOM_SDK_itomCommonLib_LIBRARY} ${ITOM_SDK_itomCommonQtLib_LIBRARY})

#handle translations BEGIN STEP 2
if (UPDATE_TRANSLATIONS)
	QT4_CREATE_TRANSLATION_ITOM(TRANSLATION_OUTPUT_FILES TRANSLATIONS_FILES ${target_name} ITOM_LANGUAGES ${FILES_TO_TRANSLATE} ${TRANSLATIONS_FILES})
	
	add_custom_target (_${target_name}_translation DEPENDS ${TRANSLATION_OUTPUT_FILES})
	add_dependencies(${target_name} _${target_name}_translation)
	
	QT4_ADD_TRANSLATION_ITOM(QM_FILES "${CMAKE_CURRENT_BINARY_DIR}/translation" ${target_name} ${TRANSLATIONS_FILES})
else (UPDATE_TRANSLATIONS)
	QT4_ADD_TRANSLATION_ITOM(QM_FILES "${CMAKE_CURRENT_BINARY_DIR}/translation" ${target_name} ${EXISTING_TRANSLATION_FILES})
endif (UPDATE_TRANSLATIONS)
#handle translations END STEP 2

# COPY SECTION
set(COPY_SOURCES "")
set(COPY_DESTINATIONS "")

# here the output library of your plugin marked for copying to itom's plugin directory in a post-build process
ADD_PLUGINLIBRARY_TO_COPY_LIST(${target_name} COPY_SOURCES COPY_DESTINATIONS)

# you can mark further files for this post-build copy process. Therefore you always need to define the source
# file(s) and the destination folder, that is created if it does not exist:

# 2. copy any file from source (EXAMPLE)
#LIST(APPEND COPY_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/matplotlibWidget.h)
#LIST(APPEND COPY_DESTINATIONS ${ITOM_DIR}/designer)

# here the translation files are set to the copy list as well
ADD_QM_FILES_TO_COPY_LIST(${target_name} QM_FILES COPY_SOURCES COPY_DESTINATIONS)

# now, all files that need to be copied in the post-build process are listed in the variable
# COPY_SOURCES whereas its destination folders are in COPY_DESTINATIONS

# the post-build process is finally created here:
POST_BUILD_COPY_FILES(${target_name} COPY_SOURCES COPY_DESTINATIONS)

