SET (target_name itomCommonLib)  
project(${target_name}) 
cmake_minimum_required(VERSION 2.8)

IF(APPLE AND CMAKE_VERSION VERSION_GREATER 2.8.7)
    if(POLICY CMP0042)
        cmake_policy(SET CMP0042 OLD)
    ENDIF(POLICY CMP0042)
ENDIF()

OPTION(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." ON) 
SET (ITOM_SDK_DIR "" CACHE PATH "base path to itom_sdk")
SET(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.")
OPTION(BUILD_UNICODE "Build with unicode charset if set to ON, else multibyte charset." ON)
OPTION(BUILD_ITOMLIBS_SHARED "Build dataObject, pointCloud, itomCommonLib and itomCommonQtLib as shared library (default)" ON)
 
SET (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/..)
       
include("../ItomBuildMacros.cmake")
    
IF(BUILD_ITOMLIBS_SHARED)
    SET(LIBRARY_TYPE SHARED)
    
    ADD_DEFINITIONS(-DITOMLIBS_SHARED -D_ITOMLIBS_SHARED)
    ADD_DEFINITIONS(-DITOMCOMMON_DLL -D_ITOMCOMMON_DLL)
ELSE(BUILD_ITOMLIBS_SHARED)
    SET(LIBRARY_TYPE STATIC)
    
    #if itomCommon is static, add -fPIC as compiler flag for linux
    IF(UNIX)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    ENDIF(UNIX)
ENDIF(BUILD_ITOMLIBS_SHARED)

IF (BUILD_UNICODE)
    ADD_DEFINITIONS(-DUNICODE -D_UNICODE)
ENDIF (BUILD_UNICODE)

# default build types are None, Debug, Release, RelWithDebInfo and MinRelSize
IF (DEFINED CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ELSE(CMAKE_BUILD_TYPE)
    SET (CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ENDIF (DEFINED CMAKE_BUILD_TYPE)

set(itomCommon_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/commonVersion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/retVal.h
    ${CMAKE_CURRENT_SOURCE_DIR}/typeDefs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/numeric.h
    ${CMAKE_CURRENT_SOURCE_DIR}/byteArray.h
    ${CMAKE_CURRENT_SOURCE_DIR}/param.h
    ${CMAKE_CURRENT_SOURCE_DIR}/paramMeta.h
    ${CMAKE_CURRENT_SOURCE_DIR}/color.h
    ${CMAKE_CURRENT_SOURCE_DIR}/helperColor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/interval.h
    ${CMAKE_CURRENT_SOURCE_DIR}/itomPlotHandle.h
    ${CMAKE_CURRENT_SOURCE_DIR}/sharedStructures.h
)

set(itomCommon_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/sources/retVal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sources/byteArray.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sources/param.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sources/paramMeta.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sources/interval.cpp
)

if(MSVC)
    list(APPEND itomCommon_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/version.rc)
endif(MSVC)

add_library(${target_name} ${LIBRARY_TYPE} ${itomCommon_SOURCES} ${itomCommon_HEADERS})

TARGET_LINK_LIBRARIES(${target_name})

# COPY SECTION
set(COPY_SOURCES "")
set(COPY_DESTINATIONS "")
ADD_LIBRARY_TO_APPDIR_AND_SDK(${target_name} COPY_SOURCES COPY_DESTINATIONS)
POST_BUILD_COPY_FILES(${target_name} COPY_SOURCES COPY_DESTINATIONS)
