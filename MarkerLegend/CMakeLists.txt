SET (target_name markerLegend)

project(${target_name})

cmake_minimum_required(VERSION 2.8)

IF(APPLE AND CMAKE_VERSION VERSION_GREATER 2.8.7)
    if(POLICY CMP0042)
        cmake_policy(SET CMP0042 OLD)
    ENDIF(POLICY CMP0042)
ENDIF()

OPTION(BUILD_UNICODE "Build with unicode charset if set to ON, else multibyte charset." ON)
OPTION(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." OFF)
SET (ITOM_SDK_DIR "" CACHE PATH "base path to itom_sdk")
SET(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.") 
OPTION(BUILD_ITOMLIBS_SHARED "Build dataObject, pointCloud, itomCommonLib and itomCommonQtLib as shared library (default)" ON)
  
IF(BUILD_ITOMLIBS_SHARED)
    SET(LIBRARY_TYPE SHARED)
    ADD_DEFINITIONS(-DITOMLIBS_SHARED -D_ITOMLIBS_SHARED)
    ADD_DEFINITIONS(-DMARKERLEGEND_DLL -D_MARKERLEGEND_DLL)
ELSE(BUILD_ITOMLIBS_SHARED)
    SET(LIBRARY_TYPE STATIC)
    
    #if itomCommon is static, add -fPIC as compiler flag for linux
    IF(UNIX)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    ENDIF(UNIX)
ENDIF(BUILD_ITOMLIBS_SHARED)
  
SET (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/..)

include("../ItomBuildMacros.cmake")

FIND_PACKAGE_QT(ON Core PrintSupport Widgets)

add_definitions(-DUNICODE -D_UNICODE)

# default build types are None, Debug, Release, RelWithDebInfo and MinRelSize
IF (DEFINED CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ELSE(CMAKE_BUILD_TYPE)
    SET (CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ENDIF (DEFINED CMAKE_BUILD_TYPE)

INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ITOM_SDK_INCLUDE_DIRS}
    )


set(markerLegend_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/defines.h
    ${CMAKE_CURRENT_SOURCE_DIR}/markerLegendWidget.h
)

set(markerLegend_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/markerLegendWidget.cpp
)

set(markerLegend_UI
    #add content
)

set(markerLegend_RCC
    #add absolute pathes to any *.qrc resource files here
)

if (QT5_FOUND)
    QT5_WRAP_UI(markerLegend__UI_MOC ${markerLegend_UI})
    QT5_ADD_RESOURCES(markerLegend__RCC_MOC ${markerLegend_RCC})
else (QT5_FOUND)
    QT4_WRAP_CPP_ITOM(markerLegend_HEADERS_MOC ${markerLegend_HEADERS})
    QT4_WRAP_UI_ITOM(markerLegend_UI_MOC ${markerLegend_UI})
    QT4_ADD_RESOURCES(markerLegend_RCC_MOC ${markerLegend_RCC})
endif (QT5_FOUND)

add_library(${target_name} ${LIBRARY_TYPE} ${markerLegend_SOURCES} ${markerLegend_HEADERS} ${markerLegend_HEADERS_MOC} ${markerLegend_UI_MOC} ${markerLegend_RCC_MOC})

TARGET_LINK_LIBRARIES(${target_name} ${QT_LIBRARIES} itomCommonLib)
IF (QT5_FOUND)
    qt5_use_modules(${target_name} ${QT_COMPONENTS})
ENDIF (QT5_FOUND)

#COPY HEADER FILES TO SDK INCLUDE DIRECTORY
SET(SOURCEFILE ${CMAKE_CURRENT_SOURCE_DIR}/../MarkerLegend)
FILE(COPY ${SOURCEFILE}/defines.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/MarkerLegend)
FILE(COPY ${SOURCEFILE}/markerLegendWidget.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/MarkerLegend)

# COPY SECTION
set(COPY_SOURCES "")
set(COPY_DESTINATIONS "")
ADD_LIBRARY_TO_APPDIR_AND_SDK(${target_name} COPY_SOURCES COPY_DESTINATIONS)
POST_BUILD_COPY_FILES(${target_name} COPY_SOURCES COPY_DESTINATIONS)