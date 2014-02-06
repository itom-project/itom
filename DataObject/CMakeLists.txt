SET (target_name dataobject)
project(${target_name}) 

cmake_minimum_required(VERSION 2.8)

OPTION(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." OFF) 
SET (ITOM_SDK_DIR "" CACHE PATH "base path to itom_sdk")
SET(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.")
OPTION(BUILD_ITOMLIBS_SHARED "Build dataObject, pointCloud, itomCommonLib and itomCommonQtLib as shared library (default)" ON)

IF(BUILD_ITOMLIBS_SHARED)
    SET(LIBRARY_TYPE SHARED)
    
    ADD_DEFINITIONS(-DITOMLIBS_SHARED -D_ITOMLIBS_SHARED)
    ADD_DEFINITIONS(-DDATAOBJECT_DLL -D_DATAOBJECT_DLL)
ELSE(BUILD_ITOMLIBS_SHARED)
    SET(LIBRARY_TYPE STATIC)
    
    #if dataObject is static, add -fPIC as compiler flag for linux
    IF(UNIX)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    ENDIF(UNIX)
ENDIF(BUILD_ITOMLIBS_SHARED)



SET (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/..)

include("../ItomBuildMacros.cmake")


find_package(OpenCV REQUIRED)

IF (BUILD_UNICODE)
    ADD_DEFINITIONS(-DUNICODE -D_UNICODE)
ENDIF (BUILD_UNICODE)

# default build types are None, Debug, Release, RelWithDebInfo and MinRelSize
IF (DEFINED CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ELSE(CMAKE_BUILD_TYPE)
    SET (CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ENDIF (DEFINED CMAKE_BUILD_TYPE)


INCLUDE_DIRECTORIES( 
    ${OpenCV_DIR}/include
)

LINK_DIRECTORIES(
    ${OpenCV_DIR}/lib
)

set(libSrcs
    ${CMAKE_CURRENT_SOURCE_DIR}/dataobj.h
    ${CMAKE_CURRENT_SOURCE_DIR}/dataobj.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dataObjectFuncs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/dataObjectFuncs.cpp
)

add_library(${target_name} ${LIBRARY_TYPE} ${libSrcs})

IF(BUILD_ITOMLIBS_SHARED)
    TARGET_LINK_LIBRARIES(${target_name} ${OpenCV_LIBS} itomCommonLib)
ENDIF(BUILD_ITOMLIBS_SHARED)

FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../DataObject/dataobj.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/DataObject)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../DataObject/dataObjectFuncs.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/DataObject)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../DataObject/readWriteLock.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/DataObject)

# COPY SECTION
set(COPY_SOURCES "")
set(COPY_DESTINATIONS "")
ADD_LIBRARY_TO_APPDIR_AND_SDK(${target_name} COPY_SOURCES COPY_DESTINATIONS)
POST_BUILD_COPY_FILES(dataobject COPY_SOURCES COPY_DESTINATIONS)
