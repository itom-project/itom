project(DataObject)

cmake_minimum_required(VERSION 2.8)

OPTION(BUILD_TARGET64 "Build for 64 bit target if set to ON or 32 bit if set to OFF." OFF) 
SET (ITOM_SDK_DIR "" CACHE PATH "base path to itom_sdk")
SET(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Adds a postfix for debug-built libraries.")

#compile dataObject as static library (always)
SET(LIBRARY_TYPE STATIC)

#if dataObject is static, add -fPIC as compiler flag for linux
IF(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
ENDIF(UNIX)

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

add_library(dataobject ${LIBRARY_TYPE} ${libSrcs})

#TARGET_LINK_LIBRARIES(dataobject)

FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../DataObject/dataobj.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/DataObject)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../DataObject/dataObjectFuncs.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/DataObject)
FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../DataObject/readWriteLock.h  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../SDK/include/DataObject)

# COPY SECTION
set(COPY_SOURCES "")
set(COPY_DESTINATIONS "")
ADD_OUTPUTLIBRARY_TO_SDK_LIB(dataobject COPY_SOURCES COPY_DESTINATIONS)
POST_BUILD_COPY_FILES(dataobject COPY_SOURCES COPY_DESTINATIONS)
